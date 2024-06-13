import streamlit as st
import math



# Helper function to pretty-print message sizes
def convert_params(params):
    if params == 0:
        return "0"
    size_name = ("", "K", "M", "B", "T", "P", "E", "Z", "Y")
    i = int(math.floor(math.log(params, 1000)))
    p = math.pow(1000, i)
    s = round(params / p, 2)
    return "%s %s" % (s, size_name[i])

# calculates the params of a model given their hparams
def calc_params(args):
    # Calculate embedding and unembedding params. If tied, re-use the same params
    if args['tied_embeddings']:
        embedding_params = args['hidden_size'] * args['vocab_size']
    else:
        embedding_params = 2 * args['hidden_size'] * args['vocab_size']
    position_embedding_params = args['hidden_size'] * args['sequence_length']
    # Each QKVO matrix is (hxh)
    # Unless using GQA/MQA which makes K/V smaller
    attention_params = int(2 * (1 + args['kv_size_ratio']) * args['num_layers'] * args['hidden_size'] * args['hidden_size'])
    # (4*2)lh from the layernorm weights and biases for each of the QKV and mlp_in layernorms, 1h for the final layernorm.
    # the extra 4lh is a mystery but we include it here
    layernorm_params = 13 * args['num_layers'] * args['hidden_size']
    #ffn_params = 12 * args['num_layers'] * args['hidden_size'] * args['hidden_size']

    if args['moe']:
        # the number of layers that are MoE. (e.g. interval is 2 for GShard)
        num_expert_layers = args['num_layers'] / args['expert_interval']
        # the number of FFN params for each MoE layer
        ffn_expert_params = 2 * args['ffn_expansion_factor'] * num_expert_layers * args['num_experts'] * args['hidden_size'] * args['hidden_size']
        # the number of FFN params for every dense layer
        ffn_dense_params = 2 * args['ffn_expansion_factor'] * (args['num_layers'] - num_expert_layers) * args['hidden_size'] * args['hidden_size']
        ffn_params = ffn_expert_params + ffn_dense_params
        # the number of gating layer params assuming it's implemented as a simple linear layer
        gating_params = num_expert_layers * args['hidden_size'] * args['num_experts']
    else:
        # two (h x [ffn_expansion_factor * h]) FFN matrices
        ffn_params = 2 * args['ffn_expansion_factor'] * args['num_layers'] * args['hidden_size'] * args['hidden_size']

    total_params = embedding_params + attention_params + ffn_params + position_embedding_params + layernorm_params

    if args['moe']:
        total_params += gating_params

    #st.write(f'Calculating number of parameters with training configuration: {args}\n')
    st.write(f'Embedding parameters: {convert_params(embedding_params)}')
    st.write(f'Attention parameters: {convert_params(attention_params)}')
    st.write(f'FFN parameters: {convert_params(ffn_params)}')
    if args['moe']:
        st.write(f'Gating parameters: {convert_params(gating_params)}')
    st.write(f'Total Params in the Model: {convert_params(total_params)}')

# Streamlit app
def main():
    st.title("Transformer Parameter Calculator")

    num_layers = st.number_input("Number of Layers (n_layers)", value=44)
    vocab_size = st.number_input("Vocab Size", value=51200)
    hidden_size = st.number_input("Embedding or Hidden Size (d_model)", value=768)
    sequence_length = st.number_input("Sequence Length", value=2048)
    st.write("The following parameters can be left with their default values in most cases")
    tied_embeddings = st.checkbox("Tied Embeddings", value=True)
    
    moe = st.checkbox("Mixture of Experts (MoE)")

    moe_params = {}
    if moe:
        with st.expander("MoE Parameters"):
            moe_params['num_experts'] = st.number_input("Number of Experts (MoE)", value=8)
            moe_params['expert_interval'] = st.number_input("Expert Interval (MoE)", value=1)
            moe_params['topk'] = st.number_input("Top k routing (MoE)", value=1)

    ffn_expansion_factor = st.number_input("FFN Expansion Factor", value=4)
    kv_size_ratio = st.number_input("KV Size Ratio", value=1.0)

    if st.button("Calculate Parameters"):
        args = {
            'vocab_size': vocab_size,
            'tied_embeddings': tied_embeddings,
            'hidden_size': hidden_size,
            'sequence_length': sequence_length,
            'num_layers': num_layers,
            'moe': moe,
            'num_experts': moe_params.get('num_experts', 8),
            'expert_interval': moe_params.get('expert_interval', 1),
            'topk': moe_params.get('topk', 1),
            'ffn_expansion_factor': ffn_expansion_factor,
            'kv_size_ratio': kv_size_ratio,
        }
        calc_params(args)



if __name__ == "__main__":
    main()
    # Footer content
    copyright_text = "© 2024 George Sarmonikas"

    # Render the footer
    st.markdown(copyright_text)

    # removes the three dots option at the top right
    st.markdown(
    """
    <style>
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    </style>
    """,
    unsafe_allow_html=True
    )
    #removes the Deploy button at the top right
    st.markdown(
    r"""
    <style>
    .stDeployButton {
            visibility: hidden;
        }
    </style>
    """, unsafe_allow_html=True
    )
