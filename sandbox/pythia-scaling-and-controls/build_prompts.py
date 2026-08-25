"""
Step 2 of the experiment: build a prompt dataset of ~600 short prompts.

Combines:
  - ~300 templated prompts across domains (story, explanation, opinion,
    factual completion, instruction-following, dialogue)
  - ~300 prompts derived from a public lightweight text source (WikiText-103)
    truncated to short prefixes, with a programmatic-template fallback if the
    dataset is not reachable.

Saves to artifacts/prompts.jsonl with fields: {"id", "prompt", "domain", "source"}.
"""
import json
import os
import random
import re

random.seed(42)

OUT_PATH = "artifacts/prompts.jsonl"
TARGET_TEMPLATED = 300
TARGET_CORPUS = 300

# ---------------------------------------------------------------------------
# 1. Templated prompts across 6 domains
# ---------------------------------------------------------------------------

FIXED_TEMPLATES = {
    "story": [
        "Once upon a time, the old sailor",
        "The main reason is",
        "In the future,",
        "She realized that",
        "He answered,",
        "Deep in the forest, the travelers found",
        "The last thing she remembered was",
        "As the sun set, the village",
        "Without warning, the door",
        "The old house had always",
        "Years later, they finally",
        "It was a dark and stormy night when",
        "The dragon slowly opened its eyes and",
        "After the war ended, the town",
        "The letter began with the words,",
        "On the day of the wedding,",
        "The detective walked into the room and",
        "By the time the train arrived,",
        "The child looked up and said,",
        "Somewhere over the mountains,",
    ],
    "explanation": [
        "The recipe works because",
        "This algorithm is efficient because",
        "Plants grow toward light because",
        "The bridge collapsed because",
        "Vaccines are effective because",
        "The economy slowed down because",
        "Ice floats on water because",
        "The experiment failed because",
        "This code runs slowly because",
        "The engine overheated because",
        "Prices increased because",
        "The signal was lost because",
        "This medicine works by",
        "The process is simple:",
        "To understand the theorem, first note that",
        "The main cause of the delay was",
        "This happens mainly because",
        "The key insight here is that",
        "In simple terms, this works because",
        "The reason this matters is that",
    ],
    "opinion": [
        "One advantage of solar power is",
        "In my opinion, the best approach is",
        "I think the most important factor is",
        "The biggest problem with this plan is",
        "Honestly, the movie was",
        "The strongest argument for this policy is",
        "What I like most about this city is",
        "The main drawback of remote work is",
        "Critics argue that the new law",
        "Overall, I believe this decision",
        "The best part of the trip was",
        "A major concern with this technology is",
        "People often overlook the fact that",
        "The most underrated benefit of exercise is",
        "It seems clear to me that",
        "The worst part about traffic is",
        "Supporters of the proposal claim",
        "A common misconception is that",
        "The real value of this book lies in",
        "Many experts agree that",
    ],
    "factual": [
        "The capital of France is",
        "Water boils at a temperature of",
        "The human heart has",
        "World War II ended in the year",
        "The largest planet in the solar system is",
        "Photosynthesis occurs in",
        "The speed of light is approximately",
        "Mount Everest is located in",
        "The chemical symbol for gold is",
        "The author of the novel was",
        "The first president of the United States was",
        "A triangle has three",
        "The Great Wall of China was built to",
        "DNA is made up of",
        "The stock market closed today with",
        "The population of the country grew to",
        "The company reported earnings of",
        "The study found that participants",
        "According to the report, the number of cases",
        "The new policy will take effect on",
    ],
    "instruction": [
        "To bake a cake, first",
        "Please summarize the following in one sentence:",
        "Explain how to change a tire by",
        "Write a short note thanking a friend for",
        "List one benefit of drinking water, such as",
        "Translate the phrase into French:",
        "To reset the router, you should",
        "Follow these steps to install the software:",
        "To improve your writing, try to",
        "Before starting the workout, remember to",
        "When cooking rice, make sure to",
        "To fix the bug, the developer needs to",
        "Complete the form by entering your",
        "To stay safe while hiking, always",
        "The instructions say to first",
        "To make coffee, you need to",
        "When packing for a trip, don't forget to",
        "To solve the equation, begin by",
        "Turn off the device before you",
        "To sign up, click the button and",
    ],
    "dialogue": [
        "\"I can't believe it,\" she said, because",
        "\"What do you mean?\" he asked, and",
        "\"Let's go,\" said Maria, before",
        "\"That's impossible,\" the scientist replied,",
        "\"I have some news,\" John began,",
        "\"Are you sure about this?\" she whispered,",
        "\"We need to talk,\" he said quietly,",
        "\"Welcome home,\" they shouted as",
        "\"I don't think that's a good idea,\" said Sam,",
        "\"Can you help me?\" the boy asked,",
        "\"This changes everything,\" muttered the captain,",
        "\"I promise I'll be there,\" she said,",
        "\"Watch out!\" he yelled, just as",
        "\"Why did you do that?\" she demanded,",
        "\"It's finally finished,\" the engineer announced,",
        "\"Are you coming with us?\" asked Liam,",
        "\"I've never seen anything like it,\" the guide said,",
        "\"Trust me on this one,\" she insisted,",
        "\"Something's wrong,\" he whispered, because",
        "\"Congratulations,\" the manager said, adding that",
    ],
}

TOPICS = [
    "climate change", "the new smartphone", "renewable energy", "the local election",
    "the ancient ruins", "artificial intelligence", "the space mission", "the housing market",
    "the football match", "the school curriculum", "the startup", "ocean pollution",
    "the music festival", "the medical trial", "the highway project", "the art exhibit",
    "the wildlife reserve", "the software update", "the trade agreement", "the volcano",
    "the immigration debate", "the new vaccine", "urban farming", "the merger",
    "the coral reefs", "the tax reform", "the robotics competition", "the drought",
    "the film adaptation", "the census data", "the border dispute", "the power outage",
    "the archaeological dig", "the fashion show", "the interest rate hike", "the satellite launch",
    "the labor strike", "the national park", "the chess tournament", "the currency exchange",
]

TEMPLATED_WITH_SLOT = [
    "The debate over {topic} continued because",
    "Experts studying {topic} discovered that",
    "The news report about {topic} explained that",
    "A recent study on {topic} found that",
    "Everyone was talking about {topic} because",
]


def build_templated_prompts(n_target):
    prompts = []
    seen = set()
    for domain, templates in FIXED_TEMPLATES.items():
        for t in templates:
            if t not in seen:
                seen.add(t)
                prompts.append({"prompt": t, "domain": domain, "source": "template_fixed"})
    # Full cross product of slot templates x topics (independent indices, no
    # aliasing between the two cycles).
    slot_combos = [
        (tpl, topic) for tpl in TEMPLATED_WITH_SLOT for topic in TOPICS
    ]
    random.shuffle(slot_combos)
    for tpl, topic in slot_combos:
        if len(prompts) >= n_target:
            break
        text = tpl.format(topic=topic)
        if text not in seen:
            seen.add(text)
            prompts.append({"prompt": text, "domain": "explanation_slot", "source": "template_slot"})
    random.shuffle(prompts)
    return prompts[:n_target]


# ---------------------------------------------------------------------------
# 2. Corpus-derived prompts (WikiText-103), with template fallback
# ---------------------------------------------------------------------------

def try_load_wikitext(n_target, tokenizer, min_tok=4, max_tok=15):
    try:
        from datasets import load_dataset
        # Non-streaming slice: fast once the parquet shard is cached locally,
        # and avoids the very slow per-row streaming iterator.
        ds = load_dataset(
            "Salesforce/wikitext", "wikitext-103-raw-v1", split="train[:20000]"
        )
    except Exception as e:
        print(f"[build_prompts] wikitext load failed: {e}")
        return []

    prompts = []
    seen = set()
    try:
        checked = 0
        for row in ds:
            checked += 1
            if len(prompts) >= n_target:
                break
            text = row["text"].strip()
            if len(text) < 20:
                continue
            # skip headers like "= Title ="
            if text.startswith("="):
                continue
            # split into sentences roughly, take the first clause
            sentences = re.split(r"(?<=[.!?])\s+", text)
            for sent in sentences:
                sent = sent.strip()
                if len(sent) < 15:
                    continue
                words = sent.split()
                if len(words) < 4:
                    continue
                # take a short prefix (roughly 4-10 words -> ~4-15 tokens)
                cut = min(len(words), random.randint(4, 9))
                prefix = " ".join(words[:cut])
                n_tok = len(tokenizer.encode(prefix))
                if n_tok < min_tok or n_tok > max_tok:
                    continue
                if prefix in seen:
                    continue
                seen.add(prefix)
                prompts.append({"prompt": prefix, "domain": "corpus_wikitext", "source": "wikitext-103-raw-v1"})
                break
            if len(prompts) >= n_target:
                break
    except Exception as e:
        print(f"[build_prompts] wikitext iteration failed after {len(prompts)} prompts: {e}")
    return prompts


def build_corpus_fallback_prompts(n_target, existing_prompts_text):
    """Extra programmatic templates if the corpus source is unavailable."""
    extra_fixed = {
        "story2": [
            "The astronaut looked back at Earth and",
            "Inside the cave, they discovered",
            "The music stopped suddenly when",
            "Halfway through the race, she",
            "The crowd fell silent as",
            "Beneath the old bridge, someone had left",
            "The storm knocked out power across",
            "At midnight, the phone rang and",
            "The scientist's hands trembled as",
            "The map led them straight to",
        ],
        "opinion2": [
            "The most surprising thing about the trip was",
            "Looking back, the decision to move was",
            "The teacher's advice turned out to be",
            "The hardest part of learning a language is",
            "The biggest lesson from the project was",
            "Compared to last year, the results were",
            "The critics were divided over whether",
            "Fans were thrilled when the band announced",
            "The committee ultimately decided that",
            "Investors reacted to the announcement by",
        ],
        "factual2": [
            "The river flows from the mountains into",
            "The satellite was launched in order to",
            "The museum's newest exhibit features",
            "The company's headquarters are located in",
            "The species is known for its ability to",
            "The treaty was signed after years of",
            "The bridge spans a distance of",
            "The comet will not be visible again until",
            "The vaccine requires two doses given",
            "The forest fire spread quickly due to",
        ],
        "instruction2": [
            "To water the plants correctly, you should",
            "Before submitting the report, check that",
            "To calm down quickly, try to",
            "When driving in the rain, remember to",
            "To organize your files, start by",
            "To improve sleep quality, avoid",
            "When negotiating a price, always",
            "To keep the kitchen clean, wipe down",
            "To train the puppy, begin with",
            "To save battery life, turn off",
        ],
        "dialogue2": [
            "\"You did great today,\" the coach said,",
            "\"Where were you last night?\" she asked,",
            "\"I found something strange,\" he whispered,",
            "\"This is the best day ever,\" the kid shouted,",
            "\"Please be careful out there,\" mom said,",
            "\"I never expected this,\" the winner admitted,",
            "\"Let's not tell anyone yet,\" she suggested,",
            "\"That was a close call,\" the pilot said,",
            "\"I'll take full responsibility,\" the manager stated,",
            "\"Everything will be fine,\" he reassured her,",
        ],
    }
    prompts = []
    for domain, templates in extra_fixed.items():
        for t in templates:
            if t not in existing_prompts_text:
                prompts.append({"prompt": t, "domain": domain, "source": "template_fallback"})
    # If still not enough, recombine topics with new templates
    more_templates = [
        "New research on {topic} suggests that",
        "The panel discussing {topic} agreed that",
        "Public opinion on {topic} shifted after",
        "The documentary about {topic} revealed that",
        "Analysts covering {topic} predicted that",
        "The report on {topic} warned that",
    ]
    idx = 0
    while len(prompts) < n_target:
        template = more_templates[idx % len(more_templates)]
        topic = TOPICS[idx % len(TOPICS)]
        text = template.format(topic=topic)
        if text not in existing_prompts_text:
            prompts.append({"prompt": text, "domain": "factual_slot", "source": "template_fallback_slot"})
        idx += 1
    return prompts[:n_target]


def main():
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained("EleutherAI/pythia-1.4b")

    templated = build_templated_prompts(TARGET_TEMPLATED)
    templated_texts = {p["prompt"] for p in templated}

    corpus_prompts = try_load_wikitext(TARGET_CORPUS, tokenizer)
    print(f"[build_prompts] got {len(corpus_prompts)} prompts from wikitext")
    corpus_prompts = [p for p in corpus_prompts if p["prompt"] not in templated_texts]

    if len(corpus_prompts) < TARGET_CORPUS:
        needed = TARGET_CORPUS - len(corpus_prompts)
        fallback = build_corpus_fallback_prompts(
            needed, templated_texts | {p["prompt"] for p in corpus_prompts}
        )
        corpus_prompts.extend(fallback)
        print(f"[build_prompts] added {len(fallback)} fallback template prompts")

    all_prompts = templated + corpus_prompts[:TARGET_CORPUS]

    # dedupe, filter by token length 4-15, and ensure no trailing EOS
    final = []
    seen = set()
    for p in all_prompts:
        text = p["prompt"].strip()
        if text in seen or not text:
            continue
        n_tok = len(tokenizer.encode(text))
        if n_tok < 3 or n_tok > 20:
            continue
        seen.add(text)
        final.append(p)

    random.shuffle(final)
    for i, p in enumerate(final):
        p["id"] = i

    os.makedirs("artifacts", exist_ok=True)
    with open(OUT_PATH, "w") as f:
        for p in final:
            f.write(json.dumps(p) + "\n")

    print(f"[build_prompts] wrote {len(final)} prompts to {OUT_PATH}")
    from collections import Counter
    print(Counter(p["domain"] for p in final))
    print(Counter(p["source"] for p in final))


if __name__ == "__main__":
    main()
