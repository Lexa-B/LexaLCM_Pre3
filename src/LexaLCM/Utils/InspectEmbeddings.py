# src/LexaLCM/Utils/InspectEmbeddings.py

def inspect_embeddings(batch_embeds, decoder, num_batches=2, num_seqs=8):
    # Ensure batch_embeds is [B, T, 1024]
    if batch_embeds.ndim == 2:
        batch_embeds = batch_embeds.unsqueeze(1)
    elif batch_embeds.ndim == 4:
        batch_embeds = batch_embeds.squeeze(1)

    batch_size, seq_len, d_model = batch_embeds.shape
    print(f"[INSPECT_MODEL] Output embeddings (Only showing first {num_seqs} concepts per batch of {num_batches} batch(es):")
    for i in range(min(num_batches, batch_size)):
        for j in range(min(seq_len, num_seqs)):
            emb = batch_embeds[i, j, :].unsqueeze(0)  # [1, 1024]
            print(f"  Decoding batch {i} seq {j}")
            decoded = decoder(emb)
            print(f"  Seq {j}: {decoded}")

