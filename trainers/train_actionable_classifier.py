import spacy
from spacy.training.example import Example
from spacy.util import minibatch
import random

TRAIN_DATA = [
    ("Developed a mobile app used by over 1,000 users", {"cats": {"Actionable": 1, "NonActionable": 0}}),
    ("Was involved in the development of a mobile app", {"cats": {"Actionable": 0, "NonActionable": 1}}),

    ("Implemented a customer feedback loop to enhance product quality", {"cats": {"Actionable": 1, "NonActionable": 0}}),
    ("Assisted with customer feedback collection", {"cats": {"Actionable": 0, "NonActionable": 1}}),

    ("Spearheaded a campus-wide recycling initiative", {"cats": {"Actionable": 1, "NonActionable": 0}}),
    ("Participated in a recycling initiative", {"cats": {"Actionable": 0, "NonActionable": 1}}),

    ("Streamlined the hiring process to reduce time-to-hire by 25%", {"cats": {"Actionable": 1, "NonActionable": 0}}),
    ("Helped with the hiring process", {"cats": {"Actionable": 0, "NonActionable": 1}}),

    ("Launched an internal newsletter to improve team communication", {"cats": {"Actionable": 1, "NonActionable": 0}}),
    ("Involved in internal communications", {"cats": {"Actionable": 0, "NonActionable": 1}}),

    ("Optimized SQL queries to improve database performance", {"cats": {"Actionable": 1, "NonActionable": 0}}),
    ("Worked on database performance", {"cats": {"Actionable": 0, "NonActionable": 1}}),

    ("Created automated test scripts to improve deployment reliability", {"cats": {"Actionable": 1, "NonActionable": 0}}),
    ("Helped test software before release", {"cats": {"Actionable": 0, "NonActionable": 1}}),

    ("Organized a 3-day coding bootcamp for 50+ students", {"cats": {"Actionable": 1, "NonActionable": 0}}),
    ("Was part of a team that ran a coding event", {"cats": {"Actionable": 0, "NonActionable": 1}}),

    ("Redesigned website UI to improve accessibility", {"cats": {"Actionable": 1, "NonActionable": 0}}),
    ("Assisted in website design", {"cats": {"Actionable": 0, "NonActionable": 1}}),

    ("Initiated a cross-departmental collaboration to boost efficiency", {"cats": {"Actionable": 1, "NonActionable": 0}}),
    ("Collaborated with other departments", {"cats": {"Actionable": 0, "NonActionable": 1}}),

    ("Led weekly team meetings to ensure project alignment", {"cats": {"Actionable": 1, "NonActionable": 0}}),
    ("Attended weekly team meetings", {"cats": {"Actionable": 0, "NonActionable": 1}}),

    ("Designed a machine learning model that increased accuracy by 15%", {"cats": {"Actionable": 1, "NonActionable": 0}}),
    ("Exposed to machine learning concepts", {"cats": {"Actionable": 0, "NonActionable": 1}}),

    ("Coordinated logistics for a 200-person conference", {"cats": {"Actionable": 1, "NonActionable": 0}}),
    ("Helped with event logistics", {"cats": {"Actionable": 0, "NonActionable": 1}}),

    ("Built a REST API for internal tools used by engineers", {"cats": {"Actionable": 1, "NonActionable": 0}}),
    ("Supported internal engineering tools", {"cats": {"Actionable": 0, "NonActionable": 1}}),

    ("Wrote technical documentation to support new feature rollout", {"cats": {"Actionable": 1, "NonActionable": 0}}),
    ("Reviewed documentation as part of the team", {"cats": {"Actionable": 0, "NonActionable": 1}}),
    
    ("Launched a new onboarding process that reduced employee ramp-up time by 30%", {"cats": {"Actionable": 1, "NonActionable": 0}}),
    ("Was part of the onboarding team", {"cats": {"Actionable": 0, "NonActionable": 1}}),

    ("Initiated weekly customer success calls that improved retention by 18%", {"cats": {"Actionable": 1, "NonActionable": 0}}),
    ("Joined calls with customers", {"cats": {"Actionable": 0, "NonActionable": 1}}),

    ("Refactored legacy codebase to improve maintainability and reduce bugs", {"cats": {"Actionable": 1, "NonActionable": 0}}),
    ("Worked with legacy code", {"cats": {"Actionable": 0, "NonActionable": 1}}),

    ("Founded a student organization with 100+ active members", {"cats": {"Actionable": 1, "NonActionable": 0}}),
    ("Participated in a student organization", {"cats": {"Actionable": 0, "NonActionable": 1}}),

    ("Developed an automated reporting system saving 5 hours weekly", {"cats": {"Actionable": 1, "NonActionable": 0}}),
    ("Assisted with generating reports", {"cats": {"Actionable": 0, "NonActionable": 1}}),

    ("Deployed a CI/CD pipeline to accelerate feature delivery", {"cats": {"Actionable": 1, "NonActionable": 0}}),
    ("Helped with CI/CD processes", {"cats": {"Actionable": 0, "NonActionable": 1}}),

    ("Negotiated vendor contracts to cut costs by 20%", {"cats": {"Actionable": 1, "NonActionable": 0}}),
    ("Worked with external vendors", {"cats": {"Actionable": 0, "NonActionable": 1}}),

    ("Piloted a mentorship program for underrepresented students", {"cats": {"Actionable": 1, "NonActionable": 0}}),
    ("Mentored students occasionally", {"cats": {"Actionable": 0, "NonActionable": 1}}),

    ("Engineered a predictive model that reduced inventory waste by 12%", {"cats": {"Actionable": 1, "NonActionable": 0}}),
    ("Learned about predictive modeling", {"cats": {"Actionable": 0, "NonActionable": 1}}),

    ("Authored and published 3 technical blog posts with 10K+ views", {"cats": {"Actionable": 1, "NonActionable": 0}}),
    ("Read technical blogs to stay updated", {"cats": {"Actionable": 0, "NonActionable": 1}})
]

nlp = spacy.blank("en")

textcat_config = {
    "model": {
        "@architectures": "spacy.TextCatEnsemble.v2",
        "linear_model": {
            "@architectures": "spacy.TextCatBOW.v3",
            "exclusive_classes": True,
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
textcat.add_label("Actionable")
textcat.add_label("NonActionable")

optimizer = nlp.begin_training()

for epoch in range(5):
    random.shuffle(TRAIN_DATA)  # Shuffle data each epoch
    losses = {}
    batches = minibatch(TRAIN_DATA, size=2)
    for batch in batches:
        examples = []
        for text, annotations in batch:
            doc = nlp.make_doc(text)
            examples.append(Example.from_dict(doc, annotations))
        nlp.update(examples, sgd=optimizer, losses=losses)
    print(f"Epoch {epoch + 1}: Loss = {losses['textcat_multilabel']:.3f}")

nlp.to_disk("actionable_classifier")
print("Model saved to 'actionable_classifier'")
