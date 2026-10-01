# The Complete Guide to Our IoT-IDS Research Project
*A clear, deep-dive explanation of what we built, what we discovered, and what to do next.*

---

## 1. The Big Picture: What Are We Building?
We are building an **Explainable Hybrid Deep Learning-Based Intrusion Detection System (IDS)** specifically designed for **IoT Networks**. 
IoT devices (like smart sensors) don't have much computational power, so they are vulnerable to cyberattacks (like DDoS, MITM, and malware). We are using a massive dataset called **Edge-IIoTset** to train a deep learning model to detect these attacks by analyzing network traffic packets.

Our core model is a **CNN-GRU**:
- **CNN (Convolutional Neural Network):** Extracts spatial features (patterns within a single packet).
- **GRU (Gated Recurrent Unit):** Designed to capture temporal features (patterns over time). 

---

## 2. The Plot Twist: The 100% Accuracy Illusion
In the early stages of our research (Phase 1), our models were achieving **100% accuracy** on binary classification (Normal vs. Attack). 

In machine learning, 100% accuracy is almost always a red flag. It means the model isn't learning what an attack looks like; it has found a shortcut to "cheat" the test. This is called **Data Leakage**.

### How did we catch the model cheating?
We used **SHAP (SHapley Additive exPlanations)**, an advanced explainable AI tool, to look inside the model's "brain." 
SHAP revealed that the model was heavily relying on specific IoT protocols—most notably, `mqtt.topic` and `dns.qry.name.len`. 

Because of the way the Edge-IIoTset dataset was constructed (stitching different network captures together), **100% of the MQTT traffic in our sample was labeled as "Normal," and specific HTTP traffic was labeled as "Attack."** The model didn't learn intrusion detection; it just learned that "MQTT = Good" and "HTTP = Bad".

### The Fix (Feature Ablation)
To force the model to actually learn network security, we ran an ablation study (E05) where we stripped away these "cheating" MQTT features (`no_mqtt` feature policy). The model is now forced to generalize based on deeper packet inspection rather than simple protocol signatures.

---

## 3. The Multiclass Problem: Solving Extreme Class Imbalance
Detecting "Attack vs. Normal" is easy, but identifying the *specific* type of attack out of 15 different classes is very hard, especially because some attacks happen rarely (extreme class imbalance).

When we trained our CNN-GRU on all 15 classes using standard loss functions (Cross-Entropy), we got an overall accuracy of **88.0%**. Sounds good, right? 
**Wrong.** We looked at a metric called **Macro-F1**, which treats all classes equally regardless of their size. Our Macro-F1 was a terrible **0.263**. The model was just ignoring the rare attacks completely to boost its overall score on the common attacks.

### The Fix (Focal Loss)
We replaced standard Cross-Entropy with **Focal Loss** (E03). Focal Loss mathematically forces the neural network to pay more attention to the rare, difficult-to-classify attacks and down-weights the easy, common ones. 

**The incredible results:**
- Overall Accuracy jumped to **91.6%**.
- Macro-F1 score exploded to **0.408** (a massive ~55% improvement in detecting minority attacks).

---

## 4. The Data Limitation: Why We Can't Use "Real" Time-Series
Since we are using a GRU (a time-series model), you might wonder why we aren't feeding it sequences of packets over time (e.g., analyzing 10 seconds of traffic at a time).

We audited the dataset and discovered a fatal flaw in the raw data: **the timestamps are scattered, and there are essentially only 2 main device IPs dominating all the traffic.** 
Because the dataset is stitched together from isolated attack scenarios, if we tried to group packets chronologically, we would accidentally separate entire attack classes into different splits (fatal data leakage). Therefore, our GRU model processes packets one by one (`seq_len=1`). It's important to document this limitation in any research paper so we don't scientifically misrepresent the GRU's capabilities on this specific dataset.

---

## 5. What You Need To Do Next (Phase 3 & 4)

Everything mentioned above is fully implemented, audited, and tested. The codebase is safe and ready. 
Here are the exact three steps you need to take to finish the project:

### Step 1: Run the Optuna Hyperparameter Tuning (E02)
Now that we know Focal Loss works best, we need to find the absolute perfect neural network architecture (how many layers, what learning rate, what dropout rate). We have built a script using **Optuna** to automate this search.

**Your Action:**
Open your terminal and run:
```bash
python run_E02_hyperparameter_tuning.py
```
*(Note: This will train 20 different models to find the best one. It will take a few hours on a CPU, but it will aggressively kill bad models early to save time. Just let it run in the background!)*

### Step 2: Model Compression (E08)
Because this is an IoT project, deploying a massive Deep Learning model onto a tiny Edge device (like a Raspberry Pi) isn't practical. 
Once Step 1 gives you the "Best Model," you need to apply **INT8 Post-Training Quantization**. This is a technique that shrinks the file size of the neural network by converting its 32-bit floating-point weights into 8-bit integers, slightly reducing accuracy but drastically reducing size.

### Step 3: Edge Benchmarking (E09)
The final slide of your presentation or paper should prove that your model is lightweight. 
You will need to write a small script that measures **CPU Inference Latency** (how many milliseconds it takes to analyze one packet) and **Memory Footprint** (how much RAM it uses). Comparing the original model to the compressed model will be the ultimate conclusion to this research!
