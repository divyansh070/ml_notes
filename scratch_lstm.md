
## Part 5: Recurrent Neural Networks (RNNs) & LSTMs

While MLPs are great for tabular data, they have a major weakness: **they have no memory**. They process each input independently. If you are predicting the next word in a sentence, or tomorrow's stock price, the network needs to remember what happened in the past.

Recurrent Neural Networks (RNNs) introduced a "hidden state" that gets passed forward in time. However, standard RNNs suffer from the **Vanishing Gradient Problem**: during backpropagation through time (BPTT), gradients shrink exponentially, meaning the network completely forgets information from many steps ago.

### 1. Long Short-Term Memory (LSTM) Networks

To solve the vanishing gradient problem, LSTMs introduced a two-track memory system and a set of mathematical "gates" that control the flow of information.

LSTMs have two parallel memory paths:
1.  **Short-Term Memory ($h_t$):** The standard hidden state, representing immediate context.
2.  **Long-Term Memory ($C_t$):** The Cell State. This is a "highway" that runs straight down the entire chain with only minor linear interactions. This allows gradients to flow backwards uninterrupted, solving the vanishing gradient problem.

![StatQuest LSTM Forward Pass](assets/statquest_lstm_forward.png)

### 2. Hands-On Math Trace: The LSTM Gates

Let's trace exactly how an LSTM processes a new input (e.g., $1.0$), using the StatQuest visualization.

#### Stage 1: The Forget Gate (What % to remember?)
The network first decides what information in the Long-Term memory is no longer relevant and should be thrown away.
*   **Input:** It takes the *current input* (1.0) and the *previous short-term memory* (1.0).
*   **Math:** It passes them through a linear layer and applies a **Sigmoid** activation function. Sigmoid squishes numbers between 0 and 1 (representing a percentage).
*   **Result:** It outputs $0.997$. It then multiplies the Long-Term memory (2.0) by $0.997$, keeping $99.7\%$ of it ($1.99$).

#### Stage 2: The Input Gate (What new info to add?)
Next, the network decides what new information from the current step is worth saving to the Long-Term memory.
*   **Create Potential Memory (Tanh):** It creates a candidate vector using the **Tanh** activation function, squishing values between -1 and 1. (Output: $0.97$).
*   **Percentage to Add (Sigmoid):** It creates another filter using Sigmoid to decide exactly what percentage of the potential memory to actually let through. (Output: $1.0$, meaning keep 100%).
*   **Math:** It multiplies the Potential Memory by the Percentage ($0.97 \cdot 1.0 = 0.97$).
*   **Result:** It *adds* this new value ($0.97$) to the Long-Term memory highway. ($1.99 + 0.97 = \mathbf{2.96}$). This is the new Long-Term Memory ($C_t$)!

#### Stage 3: The Output Gate (Update Short-Term Memory)
Finally, the network decides what the new Short-Term memory (and the actual output prediction) should be.
*   **Filter the Long-Term:** It takes the brand new Long-Term memory (2.96) and passes it through a **Tanh** function to squish it back between -1 and 1. (Output: $0.99$).
*   **Percentage to Pass On (Sigmoid):** It creates a final Sigmoid filter based on the current input and previous short-term memory. (Output: $0.99$).
*   **Result:** It multiplies them together ($0.99 \cdot 0.99 = \mathbf{0.98}$). This $0.98$ is the new Short-Term memory ($h_t$), which is passed to the next time step, and also used as the prediction for the current step.

*(BAM!!! The LSTM has successfully maintained long-term context while updating its short-term awareness).*
