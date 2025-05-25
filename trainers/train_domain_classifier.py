import spacy
from spacy.training.example import Example
from spacy.util import minibatch

TRAIN_DATA = [
    ("Led the redesign of the onboarding process, reducing churn by 15%", {"cats": {"Domain": 1, "Generic": 0}}),
    ("Initiated a peer mentorship program across departments", {"cats": {"Domain": 1, "Generic": 0}}),
    ("Improved server response time by optimizing backend logic", {"cats": {"Domain": 1, "Generic": 0}}),
    ("Developed and deployed a task management tool for internal use", {"cats": {"Domain": 1, "Generic": 0}}),
    ("Launched an internal dashboard to automate daily KPIs for executive reporting", {"cats": {"Domain": 1, "Generic": 0}}),
    ("Refactored database schema, improving query speed by 55%", {"cats": {"Domain": 1, "Generic": 0}}),
    ("Directed a team of 5 to build a machine learning pipeline for lead scoring", {"cats": {"Domain": 1, "Generic": 0}}),
    ("Led UI/UX overhaul that boosted customer satisfaction scores by 18%", {"cats": {"Domain": 1, "Generic": 0}}),
    ("Created data visualizations in Tableau that highlighted $1.2M in cost inefficiencies", {"cats": {"Domain": 1, "Generic": 0}}),
    ("Designed a landing page used in a product launch campaign with 12K signups", {"cats": {"Domain": 1, "Generic": 0}}),
    ("Created illustrations used in a UI refresh for the mobile app", {"cats": {"Domain": 1, "Generic": 0}}),
    ("Owned implementation of authentication flow using Firebase Auth", {"cats": {"Domain": 1, "Generic": 0}}),
    ("Managed weekly deployment schedule across three environments", {"cats": {"Domain": 1, "Generic": 0}}),

    ("Participated in weekly team meetings", {"cats": {"Domain": 0, "Generic": 1}}),
    ("Was involved in data entry tasks", {"cats": {"Domain": 0, "Generic": 1}}),
    ("Helped with preparing financial reports", {"cats": {"Domain": 0, "Generic": 1}}),
    ("Worked with the sales team on client outreach", {"cats": {"Domain": 0, "Generic": 1}}),
    ("Designed a new logo as part of the brand refresh project", {"cats": {"Domain": 0, "Generic": 1}}),
    ("Responsible for maintaining documentation", {"cats": {"Domain": 0, "Generic": 1}}),
    ("Attended cross-functional syncs with product and design teams", {"cats": {"Domain": 0, "Generic": 1}}),
    ("Assisted with updating Confluence documentation", {"cats": {"Domain": 0, "Generic": 1}}),
    ("Responsible for organizing meeting notes and action items", {"cats": {"Domain": 0, "Generic": 1}}),
    ("Supported onboarding activities for new hires", {"cats": {"Domain": 0, "Generic": 1}}),
    ("Participated in efforts to improve team communication", {"cats": {"Domain": 0, "Generic": 1}}),
    ("Helped with branding materials for marketing", {"cats": {"Domain": 0, "Generic": 1}}),
    ("Worked on improving the visual identity of the platform", {"cats": {"Domain": 0, "Generic": 1}}),
    ("Responsible for code reviews and team standards", {"cats": {"Domain": 0, "Generic": 1}}),
    ("Oversaw ticket triage for the support team", {"cats": {"Domain": 0, "Generic": 1}}),
]



nlp = spacy.blank("en")

textcat_config = {
    "model": {
        "@architectures": "spacy.TextCatEnsemble.v2",
        "linear_model": {
            "@architectures": "spacy.TextCatBOW.v3",
            "exclusive_classes": False,  
            "ngram_size": 1,
            "no_output_layer": False,
        },
        "tok2vec": {
            "@architectures": "spacy.Tok2Vec.v2",
            "embed": {
                "@architectures": "spacy.MultiHashEmbed.v2",
                "width": 64,
                "rows": [2000, 2000, 500, 1000, 500],
                "attrs": ["NORM", "LOWER", "PREFIX", "SUFFIX", "SHAPE"],
                "include_static_vectors": False
            },
            "encode": {
                "@architectures": "spacy.MaxoutWindowEncoder.v2",
                "width": 64,
                "window_size": 1,
                "maxout_pieces": 3,
                "depth": 2
            },
        },
    },
    "threshold": 0.7,
}

textcat = nlp.add_pipe("textcat_multilabel", config=textcat_config)

textcat.add_label("Domain-Specific")
textcat.add_label("Non-Domain-Specific")


optimizer = nlp.begin_training()

for epoch in range(5):
    losses = {}
    batches = minibatch(TRAIN_DATA, size=2)
    for batch in batches:
        examples = []
        for text, annotations in batch:
            doc = nlp.make_doc(text)
            examples.append(Example.from_dict(doc, annotations))
        nlp.update(examples, sgd=optimizer, losses=losses)
    print(f"Epoch {epoch + 1}: Loss = {losses['textcat_multilabel']:.3f}")


nlp.to_disk("domain_classifier")
print("Model saved to 'domain_classifier'")
