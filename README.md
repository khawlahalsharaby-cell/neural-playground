# 🧠 Neural Playground | مختبر الشبكات العصبية

> **Explore Neural Networks. Understand How AI Learns.**

[![HTML5](https://img.shields.io/badge/HTML5-E34F26?style=flat-square&logo=html5&logoColor=white)](https://developer.mozilla.org/en-US/docs/Web/HTML)
[![CSS3](https://img.shields.io/badge/CSS3-1572B6?style=flat-square&logo=css3&logoColor=white)](https://developer.mozilla.org/en-US/docs/Web/CSS)
[![JavaScript](https://img.shields.io/badge/JavaScript-F7DF1E?style=flat-square&logo=javascript&logoColor=black)](https://developer.mozilla.org/en-US/docs/Web/JavaScript)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg?style=flat-square)](LICENSE)

---

## 📖 About the Project

**Neural Playground** is an interactive, browser-based educational laboratory designed to demystify Artificial Neural Networks (ANNs). 

For many beginners, machine learning feels like a "black box" buried under complex mathematical notations and heavy backend frameworks. Neural Playground bridges this gap by providing a visual, hands-on environment right inside your web browser. It allows learners to peer inside a neural network, observe how signals propagate through layers, experiment with weights and biases, and understand training mechanisms—all without requiring any prior installation or backend setup.

---

## ✨ Features

- **Interactive Neural Network Simulation:** Visualizes a multi-layer neural network (Input, Hidden, and Output layers) built with HTML5 Canvas.
- **Real-Time Forward Pass:** Adjust input sliders ($X_1, X_2$) and watch live mathematical computations update instantly across nodes.
- **Deep Component Inspection:** Click on any neuron or connection line to open a smart mathematical inspector displaying activation values ($a$), weighted sums ($z$), biases ($b$), and connection contributions.
- **Signal Tracing Mode (`Trace Signal`):** Step-by-step interactive walkthrough mode with a chronological execution log to trace data flow through the network.
- **Live Training & Backpropagation:** Train a neural network directly in the browser on sample datasets using Gradient Descent, complete with a live loss curve chart and dynamic weight updates.
- **Neuro — The AI Guide:** An interactive, animated companion character providing contextual guidance, feedback, and interactive Q&A drawers.
- **Comprehensive Knowledge Quizzes & Practical Challenges:** Multi-level quizzes (Beginner, Intermediate, Advanced) and hands-on challenges backed by an XP leveling system and achievement badges saved locally (`localStorage`).
- **Interactive Glossary:** A built-in dictionary covering key AI and deep learning terminology with everyday analogies.

---

## 📸 Screenshots Gallery

Here is a visual walkthrough of the Neural Playground interface across its core educational modules:

| Welcome & Introduction | Simulation Lab Dashboard |
| :---: | :---: |
| ![Welcome Screen](h1.png)<br>*The welcoming dashboard featuring the guide Neuro.* | ![Simulation Lab](h2.png)<br>*The core neural network workspace and interactive controls.* |

| Node & Connection Inspector | Mathematical Step-by-Step Breakdown |
| :---: | :---: |
| ![Inspector Panel](h3.png)<br>*Detailed mathematical properties of selected nodes.* | ![Math Breakdown](h4.png)<br>*Step-by-step formulas and live calculations.* |

| Signal Tracing Mode (`Trace Signal`) | Chronological Execution Timeline Log |
| :---: | :---: |
| ![Signal Tracing](h5.png)<br>*Tracing data flow across layers step-by-step.* | ![Timeline Log](h6.png)<br>*Chronological log of execution steps and values.* |

| Training & Backpropagation Module | Live Loss Curve (`Loss Curve`) |
| :---: | :---: |
| ![Training Module](h7.png)<br>*Training the network on sample classification datasets.* | ![Loss Curve](h8.png)<br>*Visualizing loss reduction over training epochs.* |

| Multi-Level Knowledge Quizzes | Interactive Glossary of AI Concepts |
| :---: | :---: |
| ![Quizzes](h9.png)<br>*Multi-level tests with instant feedback and XP rewards.* | ![Glossary](h10.png)<br>*Interactive dictionary explaining core AI concepts.* |

---

## 🛠️ Tech Stack

Neural Playground is built with a lightweight, performance-first approach using pure web technologies:
- **HTML5:** Semantic structure and custom SVG graphics for the mascot and UI icons.
- **CSS3:** Modern styling featuring Glassmorphism, CSS variables, and responsive grid layouts.
- **Vanilla JavaScript (ES6+):** Pure custom implementation of forward pass math, sigmoid activation functions, backpropagation algorithms, HTML5 Canvas rendering, and state management via `localStorage`.

---

## 📁 Project Structure

```text
neural-playground/
│
├── index.html        # Main application structure, embedded CSS, and JS engine
├── h1.png - h10.png  # Interface screenshots
└── README.md         # Project documentation
