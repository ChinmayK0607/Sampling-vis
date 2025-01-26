import streamlit as st
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

@st.cache_resource
def load_model(model_name="gpt2"):
    """
    Load the tokenizer and model. We use GPT-2 for demonstration,
    but you could load a smaller model if you want faster performance.
    """
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name)
    model.eval()
    return tokenizer, model

def top_k_filtering(logits, top_k=10, filter_value=-float("Inf")):
    """
    Filter a distribution of logits using top-k sampling.
    Keep only top_k tokens with highest probability (logits),
    set the rest to -Inf.
    """
    # Top-k
    if top_k <= 0:
        return logits  # No filtering if top_k <= 0

    # Get top k
    values, indices = torch.topk(logits, top_k)
    
    # Create a mask of all tokens to filter
    mask = torch.ones_like(logits, dtype=torch.bool)
    mask[indices] = False  # mark top_k as False so we keep them
    
    # Set filtered logits to -Inf
    filtered_logits = logits.clone()
    filtered_logits[mask] = filter_value
    return filtered_logits

def top_p_filtering(logits, top_p=0.9, filter_value=-float("Inf")):
    """
    Filter a distribution of logits using nucleus (top-p) filtering.
    """
    sorted_logits, sorted_indices = torch.sort(logits, descending=True)
    cumulative_probs = torch.cumsum(torch.softmax(sorted_logits, dim=-1), dim=-1)

    # Remove tokens with cumulative probability above the threshold
    sorted_indices_to_remove = cumulative_probs > top_p
    # Shift the mask to always keep the first token above the threshold
    sorted_indices_to_remove = sorted_indices_to_remove.roll(1, dims=0)
    sorted_indices_to_remove[0] = False

    # Scatter back to the original indices
    indices_to_remove = sorted_indices[sorted_indices_to_remove]
    filtered_logits = logits.clone()
    filtered_logits[indices_to_remove] = filter_value
    return filtered_logits

def main():
    st.title("Next-Token Explorer (Greedy, Top-K, Top-P)")

    # Load model & tokenizer
    tokenizer, model = load_model("gpt2")

    # Prompt input
    prompt = st.text_input("Enter your prompt:", value="The meaning of life is")

    # Sampling strategy selection
    sampling_mode = st.selectbox(
        "Sampling Strategy",
        ["Greedy", "Top-K", "Top-P"]
    )

    # Display appropriate slider based on selection
    if sampling_mode == "Top-K":
        top_k = st.slider("Top-K value", min_value=1, max_value=100, value=10)
    elif sampling_mode == "Top-P":
        top_p = st.slider("Top-P (nucleus) value", min_value=0.0, max_value=1.0, value=0.9, step=0.01)

    # If there's a valid prompt, compute next-token distribution
    if prompt.strip():
        with torch.no_grad():
            input_ids = tokenizer.encode(prompt, return_tensors="pt")
            outputs = model(input_ids)
            next_token_logits = outputs.logits[0, -1, :]

            # Depending on the mode, filter logits
            if sampling_mode == "Greedy":
                # Greedy effectively means no filtering for the distribution
                # We'll still show top N tokens for demonstration
                filtered_logits = next_token_logits
            elif sampling_mode == "Top-K":
                filtered_logits = top_k_filtering(next_token_logits, top_k=top_k)
            else:  # Top-P
                filtered_logits = top_p_filtering(next_token_logits, top_p=top_p)

            probabilities = torch.softmax(filtered_logits, dim=-1)

            # Show top N tokens
            top_n = 10
            top_probs, top_indices = torch.topk(probabilities, top_n)

            st.subheader(f"Top {top_n} Next-Token Candidates ({sampling_mode}):")
            for rank in range(top_n):
                token_id = top_indices[rank].item()
                token_str = tokenizer.decode([token_id], clean_up_tokenization_spaces=True)
                prob = top_probs[rank].item()
                st.write(f"**{rank+1}.** `{repr(token_str)}` (prob: {prob:.4f})")

if __name__ == "__main__":
    main()
