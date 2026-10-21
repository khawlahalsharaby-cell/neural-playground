/**
 * ============================================================================
 * Neural Playground — مختبر الشبكات العصبية
 * محرك الشبكة العصبية الحقيقي + الرسوم المتحركة التفاعلية + نظام التدريب الحي
 * ============================================================================
 */

// ----------------------------------------------------------------------------
// 1. حالة التطبيق العامة والإعدادات ( الصوت، الحركة، التنقل )
// ----------------------------------------------------------------------------
let soundEnabled = true;
let reduceMotion = false;
let simMode = 'explain'; // 'explain' أو 'explore'

// مولد مؤثرات صوتية خفيف مدمج عبر Web Audio API دون ملفات خارجية
const audioCtx = typeof (window.AudioContext || window.webkitAudioContext)() !== 'undefined'
    ? new (window.AudioContext || window.webkitAudioContext)()
    : null;

function playTone(freq = 440, duration = 0.12, type = 'sine') {
    if (!soundEnabled || !audioCtx) return;
    try {
        if (audioCtx.state === 'suspended') audioCtx.resume();
        const osc = audioCtx.createOscillator();
        const gain = audioCtx.createGain();
        osc.type = type;
        osc.frequency.setValueAtTime(freq, audioCtx.currentTime);
        gain.gain.setValueAtTime(0.08, audioCtx.currentTime);
        gain.gain.exponentialRampToValueAtTime(0.001, audioCtx.currentTime + duration);
        osc.connect(gain);
        gain.connect(audioCtx.destination);
        osc.start();
        osc.stop(audioCtx.currentTime + duration);
    } catch (e) { /* تجاهل في حال حظر المتصفح للصوت التلقائي */ }
}

function toggleSound() {
    soundEnabled = !soundEnabled;
    const btn = document.getElementById('btn-sound');
    btn.textContent = soundEnabled ? '🔊' : '🔇';
    btn.classList.toggle('active-toggle', !soundEnabled);
    if (soundEnabled) playTone(520, 0.1);
}

function toggleMotion() {
    reduceMotion = !reduceMotion;
    document.body.classList.toggle('reduce-motion', reduceMotion);
    const btn = document.getElementById('btn-motion');
    btn.textContent = reduceMotion ? '🛑' : '✨';
    btn.classList.toggle('active-toggle', reduceMotion);
}

function navigateTo(pageId) {
    playTone(480, 0.08);
    document.querySelectorAll('.page-section').forEach(sec => sec.classList.remove('active'));
    document.getElementById(pageId).classList.add('active');

    const pages = ['page-intro', 'page-sim', 'page-train', 'page-quiz', 'page-glossary'];
    const idx = pages.indexOf(pageId);

    document.querySelectorAll('.nav-btn').forEach((btn, i) => {
        btn.classList.toggle('active', i === idx);
    });

    // تحديث شريط التقدم
    const progressPercent = ((idx + 1) / pages.length) * 100;
    document.getElementById('journey-bar').style.width = `${progressPercent}%`;

    // تحديث الرسوم عند فتح الصفحات
    if (pageId === 'page-sim') {
        computeForwardPass();
        drawSimCanvas();
    } else if (pageId === 'page-train') {
        updateTrainingUI();
        drawLossChart();
    }
}

function toggleMoreInfo() {
    playTone(400, 0.1);
    const panel = document.getElementById('more-info-panel');
    panel.classList.toggle('hidden');
}

// ----------------------------------------------------------------------------
// 2. المحرك الرياضي الحقيقي للشبكة العصبية (2 مدخلات -> 3 مخفية -> 1 إخراج)
// ----------------------------------------------------------------------------
function sigmoid(z) {
    return 1 / (1 + Math.exp(-z));
}

function sigmoidDerivative(a) {
    return a * (1 - a);
}

// الإعدادات الافتراضية المدروسة للشبكة في صفحة المحاكاة (تنتج توقعاً منطقياً لوجود قطة)
const defaultSimWeights = {
    // أوزان من طبقة الإدخال (2) إلى الطبقة المخفية (3): w_ih[hiddenIndex][inputIndex]
    w_ih: [
        [1.8, 1.4],   // الخلية المخفية H1: تهتم بوجود الأذنين والشوارب معاً
        [1.2, -0.6],  // الخلية المخفية H2: تركز على الأذنين أكثر
        [-0.8, 1.6]   // الخلية المخفية H3: تركز على الشوارب أكثر
    ],
    b_h: [-1.1, -0.3, -0.5], // انحيازات الطبقة المخفية
    // أوزان من الطبقة المخفية (3) إلى خلية الإخراج (1): w_ho[hiddenIndex]
    w_ho: [2.2, 0.9, 1.3],
    b_o: -1.6 // انحياز خلية الإخراج
};

// كائن الشبكة الحية المستخدم في صفحة المحاكاة
let simNet = JSON.parse(JSON.stringify(defaultSimWeights));

// حالة القيم المحسوبة حالياً في المحاكاة
let simState = {
    inputs: [0.80, 0.90],
    z_h: [0, 0, 0],
    a_h: [0, 0, 0],
    z_o: 0,
    a_o: 0
};

// تنفيذ حساب التمرير الأمامي (Forward Pass) الحقيقي
function computeForwardPass() {
    const x1 = parseFloat(document.getElementById('slider-x1').value);
    const x2 = parseFloat(document.getElementById('slider-x2').value);
    simState.inputs = [x1, x2];

    // حساب الطبقة المخفية
    for (let i = 0; i < 3; i++) {
        const z = (simNet.w_ih[i][0] * x1) + (simNet.w_ih[i][1] * x2) + simNet.b_h[i];
        simState.z_h[i] = z;
        simState.a_h[i] = sigmoid(z);
    }

    // حساب طبقة الإخراج
    let z_out = simNet.b_o;
    for (let i = 0; i < 3; i++) {
        z_out += simNet.w_ho[i] * simState.a_h[i];
    }
    simState.z_o = z_out;
    simState.a_o = sigmoid(z_out);
}

// ----------------------------------------------------------------------------
// 3. الرسم المتحرك الترحيبي في الصفحة الأولى (Intro Mini Canvas)
// ----------------------------------------------------------------------------
const introCanvas = document.getElementById('introCanvas');
const introCtx = introCanvas.getContext('2d');
let introAnimProgress = 0;

function animateIntroCanvas() {
    introCtx.clearRect(0, 0, introCanvas.width, introCanvas.height);

    const layersX = [130, 375, 620];
    const nodesY = [
        [80, 170],
        [55, 125, 195],
        [125]
    ];
    const colors = ['#38bdf8', '#818cf8', '#f472b6'];

    if (!reduceMotion) {
        introAnimProgress = (introAnimProgress + 0.008) % 1;
    } else {
        introAnimProgress = 0.5;
    }

    // رسم الخطوط والإشارات المتحركة
    for (let l = 0; l < 2; l++) {
        for (let i = 0; i < nodesY[l].length; i++) {
            for (let j = 0; j < nodesY[l + 1].length; j++) {
                const x1 = layersX[l], y1 = nodesY[l][i];
                const x2 = layersX[l + 1], y2 = nodesY[l + 1][j];

                introCtx.beginPath();
                introCtx.moveTo(x1, y1);
                introCtx.lineTo(x2, y2);
                introCtx.strokeStyle = 'rgba(148, 163, 184, 0.25)';
                introCtx.lineWidth = 2;
                introCtx.stroke();

                // إشارة متحركة
                const stageOffset = l * 0.5;
                let localP = (introAnimProgress * 2 - l);
                if (localP >= 0 && localP <= 1) {
                    const sx = x1 + (x2 - x1) * localP;
                    const sy = y1 + (y2 - y1) * localP;
                    introCtx.beginPath();
                    introCtx.arc(sx, sy, 5, 0, Math.PI * 2);
                    introCtx.fillStyle = '#fbbf24';
                    introCtx.shadowColor = '#fbbf24';
                    introCtx.shadowBlur = 10;
                    introCtx.fill();
                    introCtx.shadowBlur = 0;
                }
            }
        }
    }

    // رسم الخلايا الكرتونية اللطيفة
    for (let l = 0; l < 3; l++) {
        for (let i = 0; i < nodesY[l].length; i++) {
            const cx = layersX[l], cy = nodesY[l][i];
            const isGlowing = (l === 0 && introAnimProgress < 0.35) ||
                              (l === 1 && introAnimProgress >= 0.35 && introAnimProgress < 0.75) ||
                              (l === 2 && introAnimProgress >= 0.75);

            drawCuteNeuron(introCtx, cx, cy, 24, colors[l], isGlowing, 'happy');
        }
    }

    requestAnimationFrame(animateIntroCanvas);
}

// دالة مساعدة لرسم خلية عصبية ذات وجه كرتوني لطيف
function drawCuteNeuron(ctx, x, y, radius, color, isHighlighted, expression = 'happy', labelText = '') {
    ctx.save();
    ctx.beginPath();
    ctx.arc(x, y, radius, 0, Math.PI * 2);
    ctx.fillStyle = '#1e293b';
    ctx.fill();

    ctx.lineWidth = isHighlighted ? 4 : 2.5;
    ctx.strokeStyle = isHighlighted ? '#fbbf24' : color;
    if (isHighlighted) {
        ctx.shadowColor = color;
        ctx.shadowBlur = 18;
    }
    ctx.stroke();
    ctx.shadowBlur = 0;

    // العيون
    const eyeOffsetX = radius * 0.32;
    const eyeOffsetY = -radius * 0.15;
    ctx.fillStyle = '#ffffff';
    ctx.beginPath();
    ctx.arc(x - eyeOffsetX, y + eyeOffsetY, radius * 0.13, 0, Math.PI * 2);
    ctx.arc(x + eyeOffsetX, y + eyeOffsetY, radius * 0.13, 0, Math.PI * 2);
    ctx.fill();

    // الفم الكرتوني
    ctx.beginPath();
    ctx.strokeStyle = '#ffffff';
    ctx.lineWidth = 2;
    ctx.lineCap = 'round';
    if (expression === 'happy') {
        ctx.arc(x, y + radius * 0.05, radius * 0.28, 0.15 * Math.PI, 0.85 * Math.PI);
    } else if (expression === 'thinking') {
        ctx.moveTo(x - radius * 0.2, y + radius * 0.22);
        ctx.lineTo(x + radius * 0.2, y + radius * 0.15);
    } else {
        ctx.arc(x, y + radius * 0.18, radius * 0.14, 0, Math.PI * 2);
    }
    ctx.stroke();

    // كتابة القيمة أسفل أو أعلى الخلية إن وجدت
    if (labelText) {
        ctx.fillStyle = '#f8fafc';
        ctx.font = 'bold 12px "Fira Code", monospace';
        ctx.textAlign = 'center';
        ctx.fillText(labelText, x, y + radius + 18);
    }
    ctx.restore();
}

// ----------------------------------------------------------------------------
// 4. الصفحة الثانية: مختبر المحاكاة التفاعلية (5 مراحل مرئية + فحص العناصر)
// ----------------------------------------------------------------------------
const simCanvas = document.getElementById('simCanvas');
const simCtx = simCanvas.getContext('2d');

// إحداثيات الخلايا في كانفاس المحاكاة
const simNodes = {
    inputs: [
        { id: 'X1', name: 'خلية الإدخال X₁ (الأذنان)', x: 120, y: 145, layer: 0, idx: 0 },
        { id: 'X2', name: 'خلية الإدخال X₂ (الشوارب)', x: 120, y: 315, layer: 0, idx: 1 }
    ],
    hidden: [
        { id: 'H1', name: 'الخلية المخفية H₁', x: 380, y: 95, layer: 1, idx: 0 },
        { id: 'H2', name: 'الخلية المخفية H₂', x: 380, y: 230, layer: 1, idx: 1 },
        { id: 'H3', name: 'الخلية المخفية H₃', x: 380, y: 365, layer: 1, idx: 2 }
    ],
    output: [
        { id: 'O1', name: 'خلية الإخراج O₁ (التوقع)', x: 640, y: 230, layer: 2, idx: 0 }
    ]
};

let currentSimStage = 0; // 0: جاهز, 1: استقبال, 2: إشارات للمخفية, 3: حساب المخفية, 4: إنتاج النتيجة, 5: الملخص
let isSimPlaying = false;
let isSimPaused = false;
let signalProgress = 0; // من 0 إلى 1 لحركة الكرات المضيئة
let animFrameId = null;
let selectedElement = { type: 'node', layer: 1, idx: 0 }; // العنصر المختار في الفاحص

function setSimMode(mode) {
    simMode = mode;
    document.getElementById('mode-explain').classList.toggle('active', mode === 'explain');
    document.getElementById('mode-explore').classList.toggle('active', mode === 'explore');
    if (mode === 'explore') {
        currentSimStage = 4;
        computeForwardPass();
        updatePredictionGauge();
        updateDialogue('🧭', 'وضع الاستكشاف الحر', 'غيّري قيم المدخلات وشاهدي النتيجة تتحدث فوراً، واضغطي على أي خلية أو خط اتصال لفحص معادلته!');
        drawSimCanvas();
    } else {
        replaySimulation();
    }
}

function onInputSliderChange() {
    const x1 = parseFloat(document.getElementById('slider-x1').value);
    const x2 = parseFloat(document.getElementById('slider-x2').value);
    document.getElementById('val-x1').textContent = x1.toFixed(2);
    document.getElementById('val-x2').textContent = x2.toFixed(2);

    computeForwardPass();
    if (simMode === 'explore' || currentSimStage >= 4) {
        updatePredictionGauge();
    }
    updateInspectorView();
    drawSimCanvas();
}

function setInputs(v1, v2) {
    playTone(540, 0.08);
    document.getElementById('slider-x1').value = v1;
    document.getElementById('slider-x2').value = v2;
    onInputSliderChange();
}

function updateDialogue(avatar, stageText, speechText) {
    document.getElementById('sim-avatar').textContent = avatar;
    document.getElementById('sim-stage-badge').textContent = stageText;
    document.getElementById('sim-speech-text').innerHTML = speechText;
}

// رسم شبكة المحاكاة التفاعلية بالكامل
function drawSimCanvas() {
    simCtx.clearRect(0, 0, simCanvas.width, simCanvas.height);
    const showWeights = document.getElementById('chk-show-weights').checked;

    // 1. رسم خطوط الاتصال من الإدخال إلى المخفية
    for (let i = 0; i < 3; i++) {
        for (let j = 0; j < 2; j++) {
            const n1 = simNodes.inputs[j];
            const n2 = simNodes.hidden[i];
            const w = simNet.w_ih[i][j];
            const signalStrength = Math.abs(w * simState.inputs[j]);
            const isSelected = selectedElement && selectedElement.type === 'conn_ih' && selectedElement.i === i && selectedElement.j === j;

            drawConnectionLine(simCtx, n1.x, n1.y, n2.x, n2.y, w, signalStrength, isSelected, showWeights);

            // رسم الإشارة المتحركة في المرحلة 2
            if (currentSimStage === 2) {
                drawMovingPulse(simCtx, n1.x, n1.y, n2.x, n2.y, signalProgress, w >= 0 ? '#38bdf8' : '#f87171', signalStrength);
            }
        }
    }

    // 2. رسم خطوط الاتصال من المخفية إلى الإخراج
    for (let i = 0; i < 3; i++) {
        const n1 = simNodes.hidden[i];
        const n2 = simNodes.output[0];
        const w = simNet.w_ho[i];
        const signalStrength = Math.abs(w * simState.a_h[i]);
        const isSelected = selectedElement && selectedElement.type === 'conn_ho' && selectedElement.i === i;

        drawConnectionLine(simCtx, n1.x, n1.y, n2.x, n2.y, w, signalStrength, isSelected, showWeights);

        // رسم الإشارة المتحركة في المرحلة 4
        if (currentSimStage === 4 && isSimPlaying) {
            drawMovingPulse(simCtx, n1.x, n1.y, n2.x, n2.y, signalProgress, w >= 0 ? '#f472b6' : '#f87171', signalStrength);
        }
    }

    // 3. رسم خلايا الإدخال
    simNodes.inputs.forEach((node, idx) => {
        const val = simState.inputs[idx];
        const active = currentSimStage >= 1;
        const isSel = selectedElement && selectedElement.type === 'node' && selectedElement.layer === 0 && selectedElement.idx === idx;
        drawInteractiveNode(simCtx, node.x, node.y, 32, '#38bdf8', active || isSel, node.id, `x=${val.toFixed(2)}`, val);
    });

    // 4. رسم الخلايا المخفية
    simNodes.hidden.forEach((node, idx) => {
        const val = simState.a_h[idx];
        const active = currentSimStage >= 3;
        const isSel = selectedElement && selectedElement.type === 'node' && selectedElement.layer === 1 && selectedElement.idx === idx;
        const subLabel = active ? `a=${val.toFixed(2)}` : (showWeights ? `b=${simNet.b_h[idx].toFixed(1)}` : '');
        drawInteractiveNode(simCtx, node.x, node.y, 34, '#818cf8', active || isSel, node.id, subLabel, active ? val : 0.3);
    });

    // 5. رسم خلية الإخراج
    const outNode = simNodes.output[0];
    const outVal = simState.a_o;
    const outActive = currentSimStage >= 4;
    const isOutSel = selectedElement && selectedElement.type === 'node' && selectedElement.layer === 2;
    const outLabel = outActive ? `${(outVal * 100).toFixed(0)}%` : (showWeights ? `b=${simNet.b_o.toFixed(1)}` : '');
    drawInteractiveNode(simCtx, outNode.x, outNode.y, 38, '#f472b6', outActive || isOutSel, outNode.id, outLabel, outActive ? outVal : 0.3);
}

function drawConnectionLine(ctx, x1, y1, x2, y2, weight, strength, isSelected, showWeights) {
    ctx.save();
    ctx.beginPath();
    ctx.moveTo(x1, y1);
    ctx.lineTo(x2, y2);

    const positive = weight >= 0;
    ctx.strokeStyle = isSelected ? '#fbbf24' : (positive ? 'rgba(56, 189, 248, 0.45)' : 'rgba(248, 113, 113, 0.45)');
    ctx.lineWidth = isSelected ? 4.5 : Math.max(1.5, Math.min(5, Math.abs(weight) * 1.8));
    ctx.stroke();

    if (showWeights) {
        const mx = x1 + (x2 - x1) * 0.45;
        const my = y1 + (y2 - y1) * 0.45 - 6;
        ctx.fillStyle = isSelected ? '#fbbf24' : '#cbd5e1';
        ctx.font = '600 11px "Fira Code", monospace';
        ctx.textAlign = 'center';
        ctx.fillText(`w:${weight.toFixed(1)}`, mx, my);
    }
    ctx.restore();
}

function drawMovingPulse(ctx, x1, y1, x2, y2, progress, color, strength) {
    const px = x1 + (x2 - x1) * progress;
    const py = y1 + (y2 - y1) * progress;
    const radius = Math.max(5, Math.min(11, 5 + strength * 3));

    ctx.save();
    ctx.beginPath();
    ctx.arc(px, py, radius, 0, Math.PI * 2);
    ctx.fillStyle = '#fbbf24';
    ctx.shadowColor = color;
    ctx.shadowBlur = 16;
    ctx.fill();
    ctx.restore();
}

function drawInteractiveNode(ctx, x, y, r, baseColor, isActive, title, bottomBadge, intensity) {
    ctx.save();

    // هالة مضيئة بحسب شدة تنشيط الخلية
    if (isActive) {
        ctx.beginPath();
        ctx.arc(x, y, r + 6, 0, Math.PI * 2);
        ctx.fillStyle = baseColor;
        ctx.globalAlpha = Math.max(0.12, intensity * 0.35);
        ctx.fill();
        ctx.globalAlpha = 1.0;
    }

    const expr = intensity > 0.65 ? 'happy' : (intensity > 0.35 ? 'thinking' : 'surprised');
    drawCuteNeuron(ctx, x, y, r, baseColor, isActive, expr, '');

    // رمز الخلية أعلى الدائرة
    ctx.fillStyle = '#fbbf24';
    ctx.font = 'bold 12px "Fira Code", monospace';
    ctx.textAlign = 'center';
    ctx.fillText(title, x, y - r - 8);

    // القيمة أسفل الدائرة
    if (bottomBadge) {
        ctx.fillStyle = '#0f172a';
        ctx.strokeStyle = baseColor;
        ctx.lineWidth = 1.5;
        ctx.beginPath();
        ctx.roundRect(x - 32, y + r + 5, 64, 20, 6);
        ctx.fill();
        ctx.stroke();

        ctx.fillStyle = '#ffffff';
        ctx.font = 'bold 11px "Fira Code", monospace';
        ctx.fillText(bottomBadge, x, y + r + 19);
    }
    ctx.restore();
}

// تشغيل المحاكاة الكاملة تلقائياً عبر المراحل الخمس
function startFullSimulation() {
    playTone(580, 0.12);
    computeForwardPass();
    currentSimStage = 1;
    isSimPlaying = true;
    isSimPaused = false;
    document.getElementById('btn-pause-sim').disabled = false;
    document.getElementById('sim-summary-panel').classList.add('hidden');
    runStageLogic();
}

function stepSimulation() {
    playTone(500, 0.09);
    computeForwardPass();
    isSimPlaying = false;
    isSimPaused = false;
    document.getElementById('btn-pause-sim').disabled = true;

    currentSimStage = currentSimStage < 5 ? currentSimStage + 1 : 1;
    signalProgress = 0.6; // إظهار الإشارات في منتصف الطريق عند التنقل اليدوي
    executeStageAnnouncement(currentSimStage);
    drawSimCanvas();
}

function togglePauseSimulation() {
    if (!isSimPlaying) return;
    isSimPaused = !isSimPaused;
    document.getElementById('btn-pause-sim').textContent = isSimPaused ? 'استكمال ▶️' : 'إيقاف مؤقت ⏸️';
    if (!isSimPaused) runStageLogic();
}

function replaySimulation() {
    cancelAnimationFrame(animFrameId);
    currentSimStage = 0;
    isSimPlaying = false;
    isSimPaused = false;
    document.getElementById('btn-pause-sim').disabled = true;
    document.getElementById('btn-pause-sim').textContent = 'إيقاف مؤقت ⏸️';
    document.getElementById('pred-gauge-fill').style.width = '0%';
    document.getElementById('pred-percent-text').textContent = '-- %';
    document.getElementById('pred-caption-text').textContent = 'اضغطي «شغّلي الشبكة» لحساب التقدير الفعلي.';
    document.getElementById('sim-summary-panel').classList.add('hidden');
    updateDialogue('😊', 'الاستعداد للبدء', 'جاهزون! اضغطي <strong>«شغّلي الشبكة!»</strong> أو <strong>«خطوة بخطوة»</strong> لبدء الرحلة من المرحلة الأولى.');
    drawSimCanvas();
}

function resetNetworkToDefault() {
    playTone(350, 0.12);
    simNet = JSON.parse(JSON.stringify(defaultSimWeights));
    setInputs(0.80, 0.90);
    replaySimulation();
}

function runStageLogic() {
    if (isSimPaused || !isSimPlaying) return;

    executeStageAnnouncement(currentSimStage);
    drawSimCanvas();

    if (currentSimStage === 1 || currentSimStage === 3) {
        setTimeout(() => {
            if (!isSimPaused && isSimPlaying) {
                currentSimStage++;
                signalProgress = 0;
                runStageLogic();
            }
        }, reduceMotion ? 400 : 1700);
    } else if (currentSimStage === 2 || currentSimStage === 4) {
        animateSignalStep(() => {
            if (!isSimPaused && isSimPlaying) {
                if (currentSimStage === 4) updatePredictionGauge();
                currentSimStage++;
                runStageLogic();
            }
        });
    } else if (currentSimStage === 5) {
        isSimPlaying = false;
        document.getElementById('btn-pause-sim').disabled = true;
    }
}

function animateSignalStep(onComplete) {
    if (reduceMotion) {
        signalProgress = 1;
        drawSimCanvas();
        setTimeout(onComplete, 400);
        return;
    }

    function step() {
        if (isSimPaused || !isSimPlaying) return;
        signalProgress += 0.022;
        drawSimCanvas();
        if (signalProgress < 1) {
            animFrameId = requestAnimationFrame(step);
        } else {
            signalProgress = 1;
            onComplete();
        }
    }
    animFrameId = requestAnimationFrame(step);
}

function executeStageAnnouncement(stage) {
    const [x1, x2] = simState.inputs;
    if (stage === 1) {
        playTone(440, 0.1);
        selectedElement = { type: 'node', layer: 0, idx: 0 };
        updateDialogue('📥', 'المرحلة 1: استقبال المعلومات',
            `تضيء خلايا الإدخال الآن! استقبلنا $X_1 = ${x1.toFixed(2)}$ (الأذنان) و $X_2 = ${x2.toFixed(2)}$ (الشوارب). تقول الخلية: <strong>«وصلتنا المعلومات! الآن سنرسلها إلى الطبقة التالية.»</strong>`);
    } else if (stage === 2) {
        playTone(520, 0.1);
        updateDialogue('⚡', 'المرحلة 2: انتقال الإشارات',
            `تتحرك الإشارات الضوئية عبر خطوط الاتصال نحو الطبقة المخفية! لاحظي أن كل إشارة تُضرب في <strong>وزن الخط ($w$)</strong>؛ الأوزان الأكبر تمنح الإشارة تأثيراً أقوى.`);
    } else if (stage === 3) {
        playTone(600, 0.12);
        selectedElement = { type: 'node', layer: 1, idx: 0 };
        updateDialogue('🧮', 'المرحلة 3: الحساب داخل الخلايا المخفية',
            `كل خلية مخفية تجمع (المدخلات × الأوزان) + الانحياز $b$، ثم تطبق دالة التنشيط $\\text{Sigmoid}$. مثلاً الخلية $H_1$ أخرجت القيمة النشطة <strong>${simState.a_h[0].toFixed(3)}</strong>!`);
    } else if (stage === 4) {
        playTone(680, 0.15);
        selectedElement = { type: 'node', layer: 2, idx: 0 };
        updatePredictionGauge();
        const pct = (simState.a_o * 100).toFixed(1);
        updateDialogue('🎯', 'المرحلة 4: إنتاج النتيجة النهائية',
            `وصلت الإشارات المعالجة إلى خلية الإخراج $O_1$! <strong>تقدير الشبكة لوجود قطة هو: ${pct}%</strong> (تذكري: هذه النسبة هي ناتج النموذج الرياضي لهذا المثال، وليست ضماناً مطلقاً).`);
    } else if (stage === 5) {
        playTone(760, 0.18);
        updateDialogue('🎉', 'المرحلة 5: فهم ما حدث بالتفصيل',
            `اكتملت الدورة بنجاح! راجعي الملخص البصري بالأسفل لفهم كيف ساهمت المدخلات والأوزان في الوصول إلى القرار النهائي.`);
        renderStage5Summary();
    }
    updateInspectorView();
}

function updatePredictionGauge() {
    const pct = (simState.a_o * 100).toFixed(1);
    document.getElementById('pred-gauge-fill').style.width = `${pct}%`;
    document.getElementById('pred-percent-text').textContent = `${pct}%`;

    let msg = '';
    if (simState.a_o >= 0.7) {
        msg = `تقدير الشبكة لوجود قطة: ${pct}% 🐱 (احتمال مرتفع لأن المدخلات نشّطت الأوزان الإيجابية بقوة!)`;
    } else if (simState.a_o >= 0.4) {
        msg = `تقدير الشبكة لوجود قطة: ${pct}% 🤔 (الشبكة غير متأكدة تماماً لأن الخصائص متوسطة).`;
    } else {
        msg = `تقدير الشبكة لوجود قطة: ${pct}% 🚫 (احتمال منخفض لغياب الخصائص الأساسية).`;
    }
    document.getElementById('pred-caption-text').textContent = msg;
}

function renderStage5Summary() {
    const panel = document.getElementById('sim-summary-panel');
    const grid = document.getElementById('summary-content-grid');
    panel.classList.remove('hidden');

    const [x1, x2] = simState.inputs;
    const [a1, a2, a3] = simState.a_h;
    const outPct = (simState.a_o * 100).toFixed(1);

    grid.innerHTML = `
        <div class="summary-item">
            <strong>1. المدخلات المستقبلة:</strong>
            <p>الأذنان المدببتان $X_1 = ${x1.toFixed(2)}$<br>الشوارب البارزة $X_2 = ${x2.toFixed(2)}$</p>
        </div>
        <div class="summary-item">
            <strong>2. نواتج الطبقة المخفية:</strong>
            <p>$H_1 = ${a1.toFixed(3)}$ | $H_2 = ${a2.toFixed(3)}$ | $H_3 = ${a3.toFixed(3)}$</p>
        </div>
        <div class="summary-item">
            <strong>3. أهم وزن مؤثر:</strong>
            <p>الخلية $H_1$ تملك أعلى وزن نحو الإخراج ($w = ${simNet.w_ho[0]}$) لأنها تجمع الميزتين معاً.</p>
        </div>
        <div class="summary-item">
            <strong>4. لماذا ظهرت النتيجة ${outPct}%؟</strong>
            <p>لأن ضرب قيم الخلايا المخفية في أوزان الإخراج مع إضافة الانحياز أعطى $z = ${simState.z_o.toFixed(2)}$، وعند تمريره في دالة Sigmoid نتج ${simState.a_o.toFixed(3)} = ${outPct}\\%$.</p>
        </div>
    `;
}

function resetAndTryAgain() {
    document.getElementById('slider-x1').focus();
    replaySimulation();
}

// التفاعل بالضغط على الخلايا أو خطوط الاتصال في الكانفاس
simCanvas.addEventListener('click', (e) => {
    const rect = simCanvas.getBoundingClientRect();
    const scaleX = simCanvas.width / rect.width;
    const scaleY = simCanvas.height / rect.height;
    const mx = (e.clientX - rect.left) * scaleX;
    const my = (e.clientY - rect.top) * scaleY;

    // فحص الضغط على أي خلية عصبية
    const allNodes = [...simNodes.inputs, ...simNodes.hidden, ...simNodes.output];
    for (const node of allNodes) {
        const dist = Math.hypot(mx - node.x, my - node.y);
        if (dist <= 38) {
            playTone(620, 0.08);
            selectedElement = { type: 'node', layer: node.layer, idx: node.idx };
            updateInspectorView();
            drawSimCanvas();
            return;
        }
    }

    // فحص الضغط بالقرب من خطوط الاتصال (من الإدخال للمخفية)
    for (let i = 0; i < 3; i++) {
        for (let j = 0; j < 2; j++) {
            const n1 = simNodes.inputs[j], n2 = simNodes.hidden[i];
            if (distToSegment(mx, my, n1.x, n1.y, n2.x, n2.y) < 10) {
                playTone(460, 0.07);
                selectedElement = { type: 'conn_ih', i, j };
                updateInspectorView();
                drawSimCanvas();
                return;
            }
        }
    }

    // فحص خطوط الاتصال (من المخفية للإخراج)
    for (let i = 0; i < 3; i++) {
        const n1 = simNodes.hidden[i], n2 = simNodes.output[0];
        if (distToSegment(mx, my, n1.x, n1.y, n2.x, n2.y) < 10) {
            playTone(460, 0.07);
            selectedElement = { type: 'conn_ho', i };
            updateInspectorView();
            drawSimCanvas();
            return;
        }
    }
});

function distToSegment(px, py, x1, y1, x2, y2) {
    const l2 = (x2 - x1) ** 2 + (y2 - y1) ** 2;
    if (l2 === 0) return Math.hypot(px - x1, py - y1);
    let t = ((px - x1) * (x2 - x1) + (py - y1) * (y2 - y1)) / l2;
    t = Math.max(0, Math.min(1, t));
    return Math.hypot(px - (x1 + t * (x2 - x1)), py - (y1 + t * (y2 - y1)));
}

// تحديث نافذة الفاحص الرياضي الذكي بحسب الخلية أو الاتصال المختار
function updateInspectorView() {
    const titleEl = document.getElementById('math-node-title');
    const breakdownEl = document.getElementById('math-live-breakdown');
    if (!selectedElement) return;

    const [x1, x2] = simState.inputs;

    if (selectedElement.type === 'node') {
        const { layer, idx } = selectedElement;
        if (layer === 0) {
            const val = simState.inputs[idx];
            titleEl.textContent = `🟢 خلية الإدخال X_${idx + 1}`;
            breakdownEl.innerHTML = `
                <p><strong>دور الخلية:</strong> تستقبل القيمة الرقمية للميزة وتمررها دون تعقيد حسابي.</p>
                <p><strong>القيمة المستقبلة الحالية:</strong> <code>${val.toFixed(2)}</code></p>
                <p><strong>ما ترسله للطبقة المخفية:</strong> ترسل الرقم <code>${val.toFixed(2)}</code> عبر 3 خطوط اتصال نحو $H_1, H_2, H_3$.</p>
            `;
        } else if (layer === 1) {
            const w1 = simNet.w_ih[idx][0], w2 = simNet.w_ih[idx][1];
            const b = simNet.b_h[idx];
            const z = simState.z_h[idx];
            const a = simState.a_h[idx];
            titleEl.textContent = `🟣 الخلية المخفية H_${idx + 1}`;
            breakdownEl.innerHTML = `
                <p><strong>1. القيم المستقبلة:</strong> $x_1 = ${x1.toFixed(2)}, x_2 = ${x2.toFixed(2)}$</p>
                <p><strong>2. الأوزان والانحياز:</strong> $w_1 = ${w1.toFixed(2)}, w_2 = ${w2.toFixed(2)}, b = ${b.toFixed(2)}$</p>
                <p><strong>3. المجموع الموزون ($z$):</strong><br>
                <code dir="ltr">z = (${w1.toFixed(2)}×${x1.toFixed(2)}) + (${w2.toFixed(2)}×${x2.toFixed(2)}) + (${b.toFixed(2)}) = ${z.toFixed(3)}</code></p>
                <p><strong>4. بعد دالة التنشيط Sigmoid ($a$):</strong><br>
                <code dir="ltr">a = σ(${z.toFixed(3)}) = ${a.toFixed(3)}</code> (ترسل هذه القيمة لخلية الإخراج).</p>
            `;
        } else if (layer === 2) {
            const [a1, a2, a3] = simState.a_h;
            const [w1, w2, w3] = simNet.w_ho;
            const b = simNet.b_o;
            const z = simState.z_o;
            const a = simState.a_o;
            titleEl.textContent = `🌸 خلية الإخراج النهائية O_1`;
            breakdownEl.innerHTML = `
                <p><strong>1. القيم المستقبلة من المخفية:</strong> $a_1=${a1.toFixed(2)}, a_2=${a2.toFixed(2)}, a_3=${a3.toFixed(2)}$</p>
                <p><strong>2. أوزان الإخراج:</strong> $w_1=${w1.toFixed(2)}, w_2=${w2.toFixed(2)}, w_3=${w3.toFixed(2)}, b=${b.toFixed(2)}$</p>
                <p><strong>3. حساب $z$:</strong> <code dir="ltr">z = ${z.toFixed(3)}</code></p>
                <p><strong>4. التوقع النهائي:</strong> <code dir="ltr">a = σ(${z.toFixed(3)}) = ${a.toFixed(3)} (${(a * 100).toFixed(1)}%)</code></p>
            `;
        }
    } else if (selectedElement.type === 'conn_ih') {
        const { i, j } = selectedElement;
        const w = simNet.w_ih[i][j];
        const x = simState.inputs[j];
        titleEl.textContent = `🔗 خط اتصال: من X_${j + 1} إلى H_${i + 1}`;
        breakdownEl.innerHTML = `
            <p><strong>الوزن الحالي ($w$):</strong> <code>${w.toFixed(2)}</code></p>
            <p><strong>الإشارة المارة ($x \\cdot w$):</strong> <code dir="ltr">${x.toFixed(2)} × ${w.toFixed(2)} = ${(x * w).toFixed(3)}</code></p>
            <p><strong>التفسير:</strong> ${w >= 0 ? 'وزن موجب يدعم تنشيط الخلية القادمة.' : 'وزن سالب يثبط (يقلل) تنشيط الخلية القادمة.'}</p>
        `;
    } else if (selectedElement.type === 'conn_ho') {
        const { i } = selectedElement;
        const w = simNet.w_ho[i];
        const a = simState.a_h[i];
        titleEl.textContent = `🔗 خط اتصال: من H_${i + 1} إلى الإخراج O_1`;
        breakdownEl.innerHTML = `
            <p><strong>الوزن الحالي ($w$):</strong> <code>${w.toFixed(2)}</code></p>
            <p><strong>الإشارة المارة ($a \\cdot w$):</strong> <code dir="ltr">${a.toFixed(3)} × ${w.toFixed(2)} = ${(a * w).toFixed(3)}</code></p>
        `;
    }
}

// ----------------------------------------------------------------------------
// 5. الصفحة الثالثة: التدريب الحقيقي (Backpropagation & Gradient Descent)
// ----------------------------------------------------------------------------
const trainingDataset = [
    { name: 'قطة سيامي 🐱', x: [0.90, 0.85], y: 1 },
    { name: 'قطة شيرازي 🐱', x: [0.80, 0.95], y: 1 },
    { name: 'كلب حراسة 🐶', x: [0.15, 0.20], y: 0 },
    { name: 'جرو صغير 🐶', x: [0.25, 0.10], y: 0 }
];

// شبكة مخصصة للتدريب تبدأ بأوزان عشوائية ضعيفة ليرى المستخدم كيف تتعلم فعلياً
let trainNet = {
    w_ih: [[0.2, -0.3], [-0.4, 0.1], [0.3, -0.2]],
    b_h: [0.0, 0.1, -0.1],
    w_ho: [-0.3, 0.2, -0.1],
    b_o: 0.0
};

let trainEpochs = 0;
let lossHistory = [];
let learningRate = 0.5;

function updateLearningRate() {
    learningRate = parseFloat(document.getElementById('lr-select').value);
}

function forwardSingleSample(net, x) {
    const a_h = [];
    for (let i = 0; i < 3; i++) {
        const z = net.w_ih[i][0] * x[0] + net.w_ih[i][1] * x[1] + net.b_h[i];
        a_h.push(sigmoid(z));
    }
    let z_o = net.b_o;
    for (let i = 0; i < 3; i++) {
        z_o += net.w_ho[i] * a_h[i];
    }
    const a_o = sigmoid(z_o);
    return { a_h, a_o };
}

// تنفيذ دورة تدريب كاملة (Epoch) مع الانتشار العكسي الحقيقي (Backpropagation)
function performOneEpochMath() {
    let totalLoss = 0;

    trainingDataset.forEach(sample => {
        const { x, y } = sample;
        // 1. التمرير الأمامي
        const { a_h, a_o } = forwardSingleSample(trainNet, x);

        // 2. حساب الخطأ والخسارة (MSE)
        const error = a_o - y;
        totalLoss += 0.5 * (error ** 2);

        // 3. الانتشار العكسي (Backpropagation)
        // مشتقة الخسارة بالنسبة لـ z_o
        const delta_o = error * sigmoidDerivative(a_o);

        // مشتقة الخسارة بالنسبة للطبقة المخفية
        const delta_h = [0, 0, 0];
        for (let i = 0; i < 3; i++) {
            delta_h[i] = (delta_o * trainNet.w_ho[i]) * sigmoidDerivative(a_h[i]);
        }

        // 4. تحديث الأوزان والانحيازات (Gradient Descent)
        for (let i = 0; i < 3; i++) {
            trainNet.w_ho[i] -= learningRate * delta_o * a_h[i];
            trainNet.w_ih[i][0] -= learningRate * delta_h[i] * x[0];
            trainNet.w_ih[i][1] -= learningRate * delta_h[i] * x[1];
            trainNet.b_h[i] -= learningRate * delta_h[i];
        }
        trainNet.b_o -= learningRate * delta_o;
    });

    const avgLoss = totalLoss / trainingDataset.length;
    trainEpochs++;
    lossHistory.push(avgLoss);
    if (lossHistory.length > 120) lossHistory.shift();
    return avgLoss;
}

function trainOneStep() {
    playTone(520, 0.09);
    const loss = performOneEpochMath();
    highlightTrainingStepsSequence();
    updateTrainingUI(loss);
    drawLossChart();
}

function trainMultipleEpochs(count) {
    playTone(640, 0.15);
    let step = 0;
    const btn = document.getElementById('btn-auto-train');
    btn.disabled = true;

    function loop() {
        const loss = performOneEpochMath();
        step++;
        updateTrainingUI(loss);
        drawLossChart();
        if (step < count) {
            requestAnimationFrame(loop);
        } else {
            btn.disabled = false;
            playTone(780, 0.18);
        }
    }
    loop();
}

function randomizeWeightsForTraining() {
    playTone(340, 0.12);
    const rand = () => (Math.random() * 1.2 - 0.6);
    trainNet = {
        w_ih: [[rand(), rand()], [rand(), rand()], [rand(), rand()]],
        b_h: [rand(), rand(), rand()],
        w_ho: [rand(), rand(), rand()],
        b_o: rand()
    };
    trainEpochs = 0;
    lossHistory = [];
    updateTrainingUI();
    drawLossChart();
    document.getElementById('train-neuron-face').textContent = '😵‍💫';
    document.getElementById('train-status-title').textContent = 'تم تصفير الأوزان عشوائياً!';
    document.getElementById('train-status-msg').textContent = 'أوه! نسيت كل شيء وأصبحت أوزاني عشوائية مرة أخرى 😅 اضغطي على التدريب لأتعلم من جديد!';
}

function highlightTrainingStepsSequence() {
    const steps = document.querySelectorAll('.t-step');
    steps.forEach(s => s.classList.remove('active-step'));
    steps.forEach((s, idx) => {
        setTimeout(() => {
            steps.forEach(el => el.classList.remove('active-step'));
            s.classList.add('active-step');
        }, idx * 90);
    });
}

function updateTrainingUI(latestLoss = null) {
    // حساب الخسارة الحالية إن لم تمرر
    let currentLoss = latestLoss;
    const tbody = document.getElementById('dataset-tbody');
    tbody.innerHTML = '';

    let calcLossSum = 0;
    trainingDataset.forEach(sample => {
        const { a_o } = forwardSingleSample(trainNet, sample.x);
        calcLossSum += 0.5 * ((a_o - sample.y) ** 2);
        const isCat = sample.y === 1;
        const predPct = (a_o * 100).toFixed(1);
        const isGood = (isCat && a_o > 0.7) || (!isCat && a_o < 0.3);

        tbody.innerHTML += `
            <tr>
                <td><strong>${sample.name}</strong></td>
                <td dir="ltr">[${sample.x[0]}, ${sample.x[1]}]</td>
                <td>${isCat ? 'قطة (1.0)' : 'كلب (0.0)'}</td>
                <td style="color: ${isGood ? '#34d399' : '#fbbf24'}; font-weight:700;">
                    ${a_o.toFixed(3)} (${predPct}%) ${isGood ? '✅' : '⏳'}
                </td>
            </tr>
        `;
    });

    if (currentLoss === null) currentLoss = calcLossSum / trainingDataset.length;

    document.getElementById('metric-epochs').textContent = trainEpochs;
    document.getElementById('metric-loss').textContent = currentLoss.toFixed(4);

    // تحديث التعبيرات الكرتونية الفكاهية للخلية بحسب الخسارة
    const faceEl = document.getElementById('train-neuron-face');
    const titleEl = document.getElementById('train-status-title');
    const msgEl = document.getElementById('train-status-msg');

    if (trainEpochs === 0) {
        faceEl.textContent = '🧐';
        titleEl.textContent = 'قبل التدريب (أوزان ابتدائية غير مدربة)';
        msgEl.textContent = 'توقعاتي في الجدول الآن عشوائية ومتقاربة! اضغطي «درّبي الشبكة» لنبدأ تعديل الأوزان.';
    } else if (currentLoss > 0.11) {
        faceEl.textContent = '😅';
        titleEl.textContent = 'أوه! أخطأت هذه المرة 😅';
        msgEl.textContent = 'الخسارة لا تزال مرتفعة، لنعدّل الأوزان بالانتشار العكسي ونحاول مجدداً!';
    } else if (currentLoss > 0.03) {
        faceEl.textContent = '💪';
        titleEl.textContent = 'تحسّنت النتيجة، لكن ما زال أمامنا عمل!';
        msgEl.textContent = 'لاحظي كيف بدأ منحنى الخسارة بالهبوط وبدأت أفرق بين القطط والكلاب!';
    } else {
        faceEl.textContent = '🥳';
        titleEl.textContent = 'رائع! تعلمت الأنماط بدقة عالية! 🎉';
        msgEl.textContent = 'انخفضت الخسارة قريباً من الصفر، وأصبحت أوزاني مضبوطة لإعطاء توقعات صحيحة!';
    }

    // عرض قيم الأوزان الحية
    const wGrid = document.getElementById('weights-live-grid');
    wGrid.innerHTML = `
        <div class="w-pill">w_H1: [${trainNet.w_ih[0][0].toFixed(2)}, ${trainNet.w_ih[0][1].toFixed(2)}]</div>
        <div class="w-pill">w_H2: [${trainNet.w_ih[1][0].toFixed(2)}, ${trainNet.w_ih[1][1].toFixed(2)}]</div>
        <div class="w-pill">w_H3: [${trainNet.w_ih[2][0].toFixed(2)}, ${trainNet.w_ih[2][1].toFixed(2)}]</div>
        <div class="w-pill">w_Out: [${trainNet.w_ho.map(n => n.toFixed(2)).join(', ')}]</div>
        <div class="w-pill">b_Out: ${trainNet.b_o.toFixed(2)}</div>
    `;
}

// رسم منحنى الخسارة البياني
function drawLossChart() {
    const canvas = document.getElementById('lossCanvas');
    const ctx = canvas.getContext('2d');
    ctx.clearRect(0, 0, canvas.width, canvas.height);

    // خطوط الشبكة الخلفية
    ctx.strokeStyle = 'rgba(148, 163, 184, 0.15)';
    ctx.lineWidth = 1;
    for (let y = 30; y < canvas.height; y += 40) {
        ctx.beginPath();
        ctx.moveTo(40, y);
        ctx.lineTo(canvas.width - 15, y);
        ctx.stroke();
    }

    ctx.fillStyle = '#94a3b8';
    ctx.font = '11px "Fira Code", monospace';
    ctx.fillText('Loss', 8, 20);
    ctx.fillText('0.0', 12, canvas.height - 15);

    if (lossHistory.length === 0) {
        ctx.fillStyle = '#cbd5e1';
        ctx.font = '14px "Tajawal", sans-serif';
        ctx.textAlign = 'center';
        ctx.fillText('اضغطي على أزرار التدريب لرسم منحنى انخفاض الخطأ (Loss) مباشرةً هنا', canvas.width / 2, canvas.height / 2);
        return;
    }

    const maxLoss = Math.max(0.2, ...lossHistory);
    const plotW = canvas.width - 60;
    const plotH = canvas.height - 40;

    ctx.beginPath();
    lossHistory.forEach((val, idx) => {
        const x = 45 + (idx / Math.max(1, lossHistory.length - 1)) * plotW;
        const y = (canvas.height - 20) - (val / maxLoss) * plotH;
        if (idx === 0) ctx.moveTo(x, y);
        else ctx.lineTo(x, y);
    });
    ctx.strokeStyle = '#f472b6';
    ctx.lineWidth = 3;
    ctx.stroke();
}

// ----------------------------------------------------------------------------
// 6. الصفحة الرابعة: الاختبار التفاعلي (Quiz)
// ----------------------------------------------------------------------------
const quizQuestions = [
    {
        q: '1. ما هي الوظيفة الأساسية لـ «طبقة الإدخال» (Input Layer) في الشبكة العصبية؟',
        options: [
            'استقبال البيانات الخام (الأرقام) وتمريرها إلى الطبقة التالية.',
            'تحديث الأوزان باستخدام الانحدار التدريجي.',
            'حساب مقدار الخطأ النهائي للشبكة.',
            'إنتاج القرار النهائي للتصنيف.'
        ],
        correct: 0,
        explanation: 'أحسنتِ! طبقة الإدخال هي بوابة الشبكة التي تستقبل خصائص المثال وتمررها للخلايا المخفية.'
    },
    {
        q: '2. ما المقصود بـ «الوزن» (Weight) على خط الاتصال بين خليتين؟',
        options: [
            'عدد الخلايا الموجودة في الشبكة العصبية.',
            'رقم يحدد مدى أهمية وتأثير الإشارة المارة بين الخليتين.',
            'سرعة جهاز الكمبيوتر أثناء تشغيل الكود.',
            'حجم ملف الصورة المرفوعة.'
        ],
        correct: 1,
        explanation: 'صحيح! الوزن (w) يضرب في قيمة الإشارة؛ فكلما كبر الوزن زاد تأثير تلك المعلومة في الخلية التالية.'
    },
    {
        q: '3. ماذا تفعل دالة التنشيط (Activation Function مثل Sigmoid) داخل الخلية؟',
        options: [
            'تحذف البيانات القديمة من المتصفح.',
            'تحول المجموع الحسابي (z) إلى قيمة محددة ومناسبة (مثلاً بين 0 و 1) لتمريرها وتعلم الأنماط غير الخطية.',
            'ترسم أشكالاً كرتونية على الشاشة.',
            'توقف عمل الشبكة عند حدوث خطأ.'
        ],
        correct: 1,
        explanation: 'ممتاز! دالة التنشيط تضبط مخرجات الخلية وتمنح الشبكة القدرة على تعلم العلاقات المعقدة.'
    },
    {
        q: '4. ما الفرق الجوهري بين مرحلة «التنبؤ/الاستنتاج» (Inference) ومرحلة «التدريب» (Training)؟',
        options: [
            'لا يوجد أي فرق بينهما.',
            'في التنبؤ نحسب النتيجة باستخدام الأوزان الحالية فقط، بينما في التدريب نقارن النتيجة بالصحيحة ونعدّل الأوزان لتقليل الخطأ.',
            'التدريب يتم بدون بيانات، والتنبؤ يحتاج إلى تعديل الأوزان.',
            'التنبؤ يغير عدد طبقات الشبكة.'
        ],
        correct: 1,
        explanation: 'دقيق جداً! التمرير الأمامي وحده يعطينا توقعاً، أما التدريب فيتضمن حساب الخسارة وتحديث الأوزان.'
    },
    {
        q: '5. لماذا نستخدم خوارزمية «الانتشار العكسي» (Backpropagation) أثناء التدريب؟',
        options: [
            'لعكس ألوان الصور من الأبيض والأسود إلى الملون.',
            'لمعرفة مقدار مساهمة كل وزن في الخطأ النهائي، حتى نتمكن من تعديل الأوزان في الاتجاه الصحيح.',
            'لتشغيل الشبكة من اليسار إلى اليمين فقط.',
            'لزيادة قيمة الخسارة Loss إلى أقصى حد.'
        ],
        correct: 1,
        explanation: 'رائع! الانتشار العكسي يرجع من الإخراج نحو الطبقات السابقة ليحدد مسؤولية كل وزن عن الخطأ فنصلحه.'
    }
];

let currentQuizIdx = 0;
let quizScore = 0;

function renderQuizQuestion() {
    const qObj = quizQuestions[currentQuizIdx];
    document.getElementById('quiz-q-num').textContent = currentQuizIdx + 1;
    document.getElementById('quiz-q-total').textContent = quizQuestions.length;
    document.getElementById('quiz-current-score').textContent = quizScore;
    document.getElementById('quiz-question-text').textContent = qObj.q;

    const optsContainer = document.getElementById('quiz-options-container');
    optsContainer.innerHTML = '';

    qObj.options.forEach((optText, i) => {
        const btn = document.createElement('button');
        btn.className = 'quiz-option-btn';
        btn.textContent = optText;
        btn.onclick = () => selectQuizAnswer(i, btn);
        optsContainer.appendChild(btn);
    });

    document.getElementById('quiz-feedback-box').classList.add('hidden');
}

function selectQuizAnswer(selectedIndex, btnEl) {
    const qObj = quizQuestions[currentQuizIdx];
    const allBtns = document.querySelectorAll('.quiz-option-btn');
    allBtns.forEach(b => b.disabled = true);

    const feedbackBox = document.getElementById('quiz-feedback-box');
    const feedbackTitle = document.getElementById('quiz-feedback-title');
    const feedbackExp = document.getElementById('quiz-feedback-explanation');

    if (selectedIndex === qObj.correct) {
        playTone(720, 0.15);
        btnEl.classList.add('correct');
        quizScore++;
        document.getElementById('quiz-current-score').textContent = quizScore;
        feedbackTitle.textContent = '✅ إجابة صحيحة!';
        feedbackTitle.style.color = '#34d399';
    } else {
        playTone(260, 0.2, 'sawtooth');
        btnEl.classList.add('wrong');
        allBtns[qObj.correct].classList.add('correct');
        feedbackTitle.textContent = '❌ إجابة غير دقيقة!';
        feedbackTitle.style.color = '#f87171';
    }

    feedbackExp.textContent = qObj.explanation;
    feedbackBox.classList.remove('hidden');
    document.getElementById('btn-next-question').textContent =
        currentQuizIdx < quizQuestions.length - 1 ? 'السؤال التالي ⬅️' : 'عرض النتيجة النهائية 🏆';
}

function nextQuizQuestion() {
    if (currentQuizIdx < quizQuestions.length - 1) {
        currentQuizIdx++;
        renderQuizQuestion();
    } else {
        showQuizFinalResults();
    }
}

function showQuizFinalResults() {
    document.getElementById('quiz-question-View').classList.add('hidden');
    const resView = document.getElementById('quiz-result-view');
    resView.classList.remove('hidden');

    document.getElementById('quiz-final-score').textContent = `${quizScore} / ${quizQuestions.length}`;
    const emojiEl = document.getElementById('quiz-final-emoji');
    const titleEl = document.getElementById('quiz-final-title');
    const msgEl = document.getElementById('quiz-final-message');

    if (quizScore === 5) {
        emojiEl.textContent = '🌟🏆';
        titleEl.textContent = 'عبقرية في الشبكات العصبية!';
        msgEl.textContent = 'إجاباتك كاملة! لقد فهمتِ كيف تمر الإشارات وكيف تتعلم الشبكة من أخطائها بدقة علمية ممتازة.';
    } else if (quizScore >= 3) {
        emojiEl.textContent = '👏😊';
        titleEl.textContent = 'مستوى رائع جداً!';
        msgEl.textContent = 'لديكِ أساس قوي جداً في فهم الشبكات العصبية، ومع جولة سريعة في المختبر ستتقنين كل التفاصيل!';
    } else {
        emojiEl.textContent = '🌱💡';
        titleEl.textContent = 'بداية طيبة!';
        msgEl.textContent = 'لا بأس أبداً! جربي العودة إلى مختبر المحاكاة والقاموس التفاعلي ثم أعيدي الاختبار وستلاحظين الفرق.';
    }
}

function restartQuiz() {
    currentQuizIdx = 0;
    quizScore = 0;
    document.getElementById('quiz-result-view').classList.add('hidden');
    document.getElementById('quiz-question-View').classList.remove('hidden');
    renderQuizQuestion();
}

// ----------------------------------------------------------------------------
// 7. الصفحة الخامسة: القاموس التفاعلي للمصطلحات الـ 12 (Glossary)
// ----------------------------------------------------------------------------
const glossaryTerms = [
    {
        en: 'Artificial Neural Network',
        ar: 'الشبكة العصبية الاصطناعية',
        desc: 'نموذج حاسوبي رياضي مستوحى بشكل مبسط من طريقة ترابط الخلايا، يتكون من وحدات حسابية متصلة في طبقات لاكتشاف الأنماط في البيانات.',
        example: 'تشبه فريق عمل في شركة: قسم يستقبل الطلبات، قسم يحللها، ومدير يصدر القرار النهائي.',
        visual: '[مدخلات X] ──► (خلايا مخفية H) ──► [التوقع Y]'
    },
    {
        en: 'Neuron',
        ar: 'الخلية العصبية (العقدة الحسابية)',
        desc: 'الوحدة الأساسية في الشبكة؛ تستقبل أرقاماً، تضربها في أوزانها، تضيف الانحياز، وتمرر الناتج عبر دالة تنشيط.',
        example: 'مثل حكم في مسابقة يجمع نقاط المتسابق من عدة معايير ليخرج بتقييم موحد.',
        visual: 'Input(x) ──×Weight(w)──► [ Σ + b ──► σ(z) ] ──► Output(a)'
    },
    {
        en: 'Input Layer',
        ar: 'طبقة الإدخال',
        desc: 'أول طبقة في الشبكة، مهمتها استلام الخصائص الرقمية للمثال (Features) وتمريرها للطبقة التالية دون تعديل.',
        example: 'مثل عيوننا وأذنينا التي تلتقط الصورة والصوت وترسلهما للدماغ.',
        visual: 'X₁ = 0.80  |  X₂ = 0.90  ──► تمرير مباشر للطبقة المخفية'
    },
    {
        en: 'Hidden Layer',
        ar: 'الطبقة المخفية',
        desc: 'الطبقة (أو الطبقات) الوسطى بين الإدخال والإخراج، وفيها تتم أغلب الحسابات واستخراج الأنماط المركبة.',
        example: 'مثل المطبخ داخل المطعم؛ الزبون يرى قائمة الطعام (الإدخال) والطبق الجاهز (الإخراج) ولا يرى تفاصيل الطبخ بالداخل.',
        visual: 'Input ──► [ Hidden Layer: استخلاص الملامح ] ──► Output'
    },
    {
        en: 'Output Layer',
        ar: 'طبقة الإخراج',
        desc: 'المحطة الأخيرة في الشبكة العصبية التي تعطينا النتيجة النهائية أو نسبة التوقع المطلوبة.',
        example: 'مثل شاشة الكاشير التي تعرض المبلغ النهائي بعد جمع كافة الأصناف والضرائب.',
        visual: 'Hidden Signals ──► [ Output Node ] ──► 76% قطة 🐱'
    },
    {
        en: 'Weights',
        ar: 'الأوزان (المعاملات)',
        desc: 'أرقام قابلة للتعديل توضع على خطوط الاتصال، وتحدد مدى قوة أو أهمية كل مدخل في التأثير على الخلية التالية.',
        example: 'عند شراء لابتوب للبرمجة، قد تعطين "المعالج" وزناً وأهمية أعلى بكثير من "لون الحقيبة"!',
        visual: 'وزن كبير (w = +2.2) = تأثير قوي ⚡ | وزن قريب من الصفر = تأثير ضعيف'
    },
    {
        en: 'Bias',
        ar: 'الانحياز',
        desc: 'رقم إضافي يُجمع مع حاصل ضرب المدخلات في الأوزان، يساعد الخلية على ضبط عتبة تنشيطها حتى لو كانت المدخلات صغيرة.',
        example: 'مثل درجات الرأفة أو النقاط الأساسية التي تُضاف للطالب لرفع أو خفض حد النجاح.',
        visual: 'z = (w₁·x₁ + w₂·x₂) + b  ◄── يزيح النتيجة يميناً أو يساراً'
    },
    {
        en: 'Activation Function',
        ar: 'دالة التنشيط',
        desc: 'معادلة رياضية تطبق على مجموع الخلية (مثل دالة Sigmoid أو ReLU) لتحويل الرقم إلى نطاق محدد وتمكين الشبكة من تعلم الأنماط المعقدة.',
        example: 'مثل منظم الإضاءة (Dimmer) الذي يحول التيار الكهربائي إلى درجة سطوع متدرجة بين 0% و 100%.',
        visual: 'Sigmoid: σ(z) = 1 / (1 + e⁻ᶻ)   ∈ (0, 1)'
    },
    {
        en: 'Loss Function',
        ar: 'دالة الخسارة (مقدار الخطأ)',
        desc: 'معادلة تقيس الفرق بين توقع الشبكة والإجابة الصحيحة؛ كلما كانت الخسارة كبيرة فهذا يعني أن الشبكة أخطأت كثيراً.',
        example: 'مثل قياس المسافة بين مكان سقوط السهم ومركز لوحة التصويب؛ هدفنا تقليص هذه المسافة للصفر!',
        visual: 'Loss = ½ (التوقع - الإجابة الصحيحة)²'
    },
    {
        en: 'Backpropagation',
        ar: 'الانتشار العكسي',
        desc: 'الآلية التي ترجع بالخطأ من طبقة الإخراج إلى الطبقات السابقة لحساب مدى مسؤولية كل وزن عن هذا الخطأ.',
        example: 'مثل مراجعة ورقة الامتحان بعد التصحيح لمعرفة أي سؤال تحديداً تسبب في نقص الدرجات.',
        visual: 'Input ◄── [توزيع المسؤولية عن الخطأ ∂Loss/∂w] ◄── Output Error'
    },
    {
        en: 'Gradient Descent',
        ar: 'الانحدار التدريجي',
        desc: 'خوارزمية التحسين التي تستخدم نتائج الانتشار العكسي لتعديل الأوزان خطوة بخطوة في الاتجاه الذي يخفض الخسارة.',
        example: 'كشخص ينزل من قمة جبل في الضباب، يتحسس ميلان الأرض تحت قدميه ويخطو دائماً نحو الأسفل حتى يصل للوادي.',
        visual: 'الوزن الجديد = الوزن القديم - (سرعة التعلم η × ميل الخطأ)'
    },
    {
        en: 'Training vs Inference',
        ar: 'التدريب مقابل الاستنتاج (التنبؤ)',
        desc: 'التدريب هو مرحلة الدراسة وتعديل الأوزان من الأمثلة، أما الاستنتاج فهو استخدام الشبكة بعد تخرجها لتوقع أمثلة جديدة دون تغيير أوزانها.',
        example: 'التدريب هو مذاكرتكِ طوال الفصل الدراسي، والاستنتاج هو إجابتكِ السريعة في يوم الاختبار أو العمل الفعلي!',
        visual: 'Training: (تعديل مستمر للأوزان) 🔄 | Inference: (أوزان ثابتة ──► توقع فوري) ⚡'
    }
];

function initGlossary() {
    const listContainer = document.getElementById('glossary-buttons-list');
    listContainer.innerHTML = '';

    glossaryTerms.forEach((item, index) => {
        const btn = document.createElement('button');
        btn.className = 'term-item-btn' + (index === 0 ? ' active' : '');
        btn.innerHTML = `
            <span>${item.ar}</span>
            <small class="en-term">${item.en}</small>
        `;
        btn.onclick = () => showGlossaryTerm(index, btn);
        listContainer.appendChild(btn);
    });

    showGlossaryTerm(0, listContainer.firstChild);
}

function showGlossaryTerm(idx, btnEl) {
    playTone(490, 0.06);
    document.querySelectorAll('.term-item-btn').forEach(b => b.classList.remove('active'));
    if (btnEl) btnEl.classList.add('active');

    const item = glossaryTerms[idx];
    const card = document.getElementById('glossary-detail-card');
    card.innerHTML = `
        <span class="detail-badge">${item.en}</span>
        <h2>${item.ar}</h2>
        <p style="font-size: 1.1rem; margin-top: 12px;">${item.desc}</p>

        <div class="detail-example-box">
            <strong>💡 مثال من الحياة اليومية:</strong>
            <p>${item.example}</p>
        </div>

        <div class="detail-visual-box" dir="ltr">
            ${item.visual}
        </div>
    `;
}

// ----------------------------------------------------------------------------
// 8. تهيئة التطبيق عند تحميل الصفحة
// ----------------------------------------------------------------------------
window.addEventListener('DOMContentLoaded', () => {
    animateIntroCanvas();
    computeForwardPass();
    updateInspectorView();
    drawSimCanvas();
    updateTrainingUI();
    drawLossChart();
    renderQuizQuestion();
    initGlossary();
});