import sys
import os
import torch
from torch.amp import autocast
from safetensors.torch import load_file

# Add the project root to Python path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../"))
sys.path.insert(0, project_root)

from Submodules.Pipeline_SONAR.src.pipelines import TextToEmbeddingPipeline, EmbeddingToTextPipeline
from LexaLCM.LCM_Model import LexaLCM

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load model
checkpoint_path = "/home/lexa/DevProjects/_Models/LexaLCM_Pre2/outputs/checkpoint-250000"
model = LexaLCM.from_pretrained(checkpoint_path)
model.eval().cuda()

# Init pipelines
encoder = TextToEmbeddingPipeline(language="eng_Latn", verbose=True, dtype=torch.float32)
decoder = EmbeddingToTextPipeline(language="eng_Latn", verbose=True, dtype=torch.float32)

# ## Input prompt as sequence of sentences
TestPrompt_0 = [
    "[[Start of Text.]]", 
    "Japan's long history is divided into many distinct periods, each contributing to the country and culture in their own way.", 
    "The Sengoku era was a period of great conflict in Japan.", 
    "Many clans and their samurai from all over Japan fought in that time.", 
    "The fighting lasted for many decades, but it was ultimately brought to an end in the unification of Japan.", 
    "It was followed by a period of peace and cultural growth.", 
    "This period was known as the Edo period.", 
    "This period brought forward many new forms of art and culture.", 
    "These include forms such as ukiyo-e woodblock paintings, kabuki theater, and haiku poetry.", 
    "The impacts of the Edo period are large and still felt in the present day cultural landscape."
     ]

TestPrompt_1 = [
    "[[Start of Text.]]", 
    "The latter half of the twentieth century was marked by significant advancements in social justice across the globe.",
    "The civil rights movement in the United States led to landmark legislation such as the Civil Rights Act of 1964 and the Voting Rights Act of 1965.",
    "Anti-apartheid activists in South Africa fought tirelessly for the end of institutionalized racial segregation, culminating in the dismantling of apartheid by the early 1990s.",
    "Women’s rights movements gained momentum worldwide, resulting in expanded access to education, employment, and political representation.",
    "Second-wave feminism in the 1960s and 1970s addressed issues like reproductive rights, workplace equality, and legal protections against discrimination.",
    "The LGBTQ+ rights movement began to organize more visibly in the wake of events like the Stonewall Riots of 1969.",
    "In many countries, decolonization efforts brought independence and self-determination to formerly colonized nations.",
    "Disability rights activists advocated for accessibility and equal treatment, leading to legislation such as the Americans with Disabilities Act of 1990.",
    "International organizations like the United Nations promoted human rights through declarations and treaties.",
    "Grassroots movements and community organizers played a crucial role in raising awareness and driving policy change.",
    "Despite substantial progress, social justice efforts often faced resistance and setbacks from established power structures.",
    "The closing decades of the century set the stage for continued advocacy and evolving definitions of equality in the twenty-first century."
]

TestPrompt_2 = [
    "[[Start of Text.]]", 
    "The 1980s marked a pivotal era in international relations, characterized by shifting alliances and renewed tensions.",
    "The Cold War rivalry between the United States and the Soviet Union dominated the global political landscape.",
    "NATO and the Warsaw Pact continued to serve as military counterweights in Europe.",
    "The arms race intensified, with both superpowers expanding their nuclear arsenals.",
    "At the same time, discussions on arms control, such as the Strategic Arms Reduction Talks, offered glimpses of potential cooperation.",
    "The Soviet Union underwent significant leadership changes, most notably with Mikhail Gorbachev rising to power in 1985.",
    "Gorbachev introduced reforms like glasnost and perestroika, which aimed to modernize the Soviet system and ease internal tensions.",
    "In the United States, President Ronald Reagan adopted a firm stance against communism, advocating for increased defense spending and strategic initiatives.",
    "Conflicts in regions like Afghanistan, Central America, and Africa became proxy battlegrounds for the superpowers.",
    "International organizations, including the United Nations, played a role in mediating regional disputes.",
    "The decade concluded with a dramatic thaw in East-West relations, setting the stage for significant changes in the following years.",
    "Many historians view the late 1980s as the beginning of the end for the Cold War."
]

TestPrompt_3 = [
    "[[Start of Text.]]", 
    "The early 21st century witnessed a surge in the popularity of Korean idol groups, marking a new era in the music industry.",
    "Entertainment companies such as SM, YG, and JYP played a significant role in shaping the idol training system.",
    "The rigorous training programs prepared young artists in singing, dancing, and public image management.",
    "Groups like TVXQ, Super Junior, and Girls' Generation became household names in South Korea.",
    "The Hallyu Wave, or Korean Wave, began to gain momentum as these groups attracted international attention.",
    "Music shows and variety programs provided essential platforms for idols to connect with fans and showcase their talents.",
    "Fan culture grew rapidly, with dedicated fandoms organizing support events and online communities.",
    "The success of idol groups in Japan and other Asian markets demonstrated the expanding reach of K-pop.",
    "Digital platforms such as YouTube played a pivotal role in spreading Korean music to a global audience.",
    "The aesthetic and performance standards set by early 21st century idol groups influenced newer generations of artists.",
    "Despite intense competition and demanding schedules, many idols maintained strong public personas.",
    "By the end of the decade, Korean idol groups had established a significant presence in the international music scene."
]

prompt = TestPrompt_0

# Encode each sentence and build autoregressive input
with torch.no_grad():
    context_embeddings = []
    for sentence in prompt:
        emb = encoder(sentence)  # shape: [1, 1024]
        emb = emb.to(torch.float32)
        context_embeddings.append(emb)

    # Stack to shape [1, T, 1024]
    context = torch.stack(context_embeddings, dim=0).unsqueeze(0).to(device)  # [1, T, 1024]

    print(f"→ Context shape: {context.shape}, dtype: {context.dtype}")

    # context_input = context[:, :-1, :]  # [1, T-1, 1024]
    # target = context[:, 1:, :]          # [1, T-1, 1024]

    # with autocast(dtype=torch.bfloat16, device_type="cuda"):
    #     pred = model(context_input)     # [1, T-1, 1024]

    # print(f"→ Output shape: {pred.shape}, dtype: {pred.dtype}")
    # print("→ Prediction vector sample:", pred.squeeze(1)[0, :10])

    # # Decode the last predicted embedding
    # decoded_Sonar = decoder(target[:, -1, :].squeeze(0))
    # decoded_LCM = decoder(pred[:, -1, :])
    # print(f"→ Decoded text - Last SONAR Embedding: {decoded_Sonar}")
    # print(f"→ Decoded text - Last LCM Embedding: {decoded_LCM}")

    # # Load EoT embedding
    # eot_path = "src/LexaLCM/Data/SpecialConcepts/EndOfText.safetensors"
    # eot_tensor = load_file(eot_path)["embedding"]  # Adjust the key if needed!
    # eot_tensor = eot_tensor.to(pred.device)  # [1, 1024]

    # # Grab the last timestep of pred and target (shape [1, 1024])
    # pred_vec = pred[:, -1, :]   # shape [1, 1024]
    # target_vec = target[:, -1, :] # shape [1, 1024]

    # # Compute L2 distances
    # def l2_dist(a, b):
    #     # both a and b are [1, 1024]
    #     return torch.norm(a - b, p=2).item()

    # l2_pred_eot = l2_dist(pred_vec, eot_tensor)
    # l2_true_eot = l2_dist(target_vec, eot_tensor)
    # l2_pred_true = l2_dist(pred_vec, target_vec)

    # print(f"→ L2 distance (Predicted vs. EoT): {l2_pred_eot:.4f}")
    # print(f"→ L2 distance (Ground Truth vs. EoT): {l2_true_eot:.4f}")
    # print(f"→ L2 distance (Predicted vs. Ground Truth): {l2_pred_true:.4f}")



    # context_input = context[:, :-1, :]  # [1, T-1, 1024]
    # target = context[:, 1:, :]          # [1, T-1, 1024]

    # with autocast(dtype=torch.bfloat16, device_type="cuda"):
    #     pred = model(context_input)     # [1, T-1, 1024]

    # print(f"→ Output shape: {pred.shape}, dtype: {pred.dtype}")

    # for t in range(pred.shape[1]):
    #     decoded_pred = decoder(pred[:, t, :])
    #     decoded_gt = decoder(target[:, t, :])
    #     print(f"Step {t}:")
    #     print(f"  → Model prediction: {decoded_pred}")
    #     print(f"  → Ground Truth:     {decoded_gt}")

    # # You can also print L2 distance at each step, if desired:
    # l2s = torch.norm(pred - target, dim=-1).squeeze(0)  # [T-1]
    # print("L2 distance at each step:", l2s.cpu().numpy())


with torch.no_grad():
    with autocast(dtype=torch.bfloat16, device_type="cuda"):
        pred = model(context)  # [1, 4, 1024]

# # Now: pred[:, -1, :] is the model's guess for the next embedding after the paragraph
# predicted_next_embedding = pred[:, -1, :]
# decoded_next_sentence = decoder(predicted_next_embedding)
# print("Model's prediction for the next sentence:", decoded_next_sentence)

# Optionally, print every prediction at each step (should match ground truth for S1, S2, S3, and then new guess for S4)
for t in range(pred.shape[1]):
    decoded = decoder(pred[:, t, :])
    print(f"Step {t} model next-token guess: {decoded}")