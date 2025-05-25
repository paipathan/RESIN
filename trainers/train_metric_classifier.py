import spacy
from spacy.training.example import Example
from spacy.util import minibatch

TRAIN_DATA = [
    ("Increased web traffic by 150% over three months", {"cats": {"Metrics": 1, "Vague": 0}}),
    ("Helped with website improvements", {"cats": {"Metrics": 0, "Vague": 1}}),
    ("Boosted email click-through rate by 27%", {"cats": {"Metrics": 1, "Vague": 0}}),
    ("Was responsible for technical tasks", {"cats": {"Metrics": 0, "Vague": 1}}),
    ("Resolved 95% of support tickets within 24 hours", {"cats": {"Metrics": 1, "Vague": 0}}),
    ("Worked on various backend issues", {"cats": {"Metrics": 0, "Vague": 1}}),
    ("Saved $15,000 per year by automating reports", {"cats": {"Metrics": 1, "Vague": 0}}),
    ("Assisted team with financial operations", {"cats": {"Metrics": 0, "Vague": 1}}),
    ("Reduced churn rate from 10% to 4% using personalized emails", {"cats": {"Metrics": 1, "Vague": 0}}),
    ("Worked in a fast-paced environment", {"cats": {"Metrics": 0, "Vague": 1}}),

    ("Created 20+ blog articles, generating 10,000 monthly views", {"cats": {"Metrics": 1, "Vague": 0}}),
    ("Performed data analysis tasks", {"cats": {"Metrics": 0, "Vague": 1}}),
    ("Improved mobile app load time by 40%", {"cats": {"Metrics": 1, "Vague": 0}}),
    ("Helped with mobile app performance", {"cats": {"Metrics": 0, "Vague": 1}}),
    ("Increased donations by $5,000 through email outreach", {"cats": {"Metrics": 1, "Vague": 0}}),
    ("Supported community fundraising efforts", {"cats": {"Metrics": 0, "Vague": 1}}),
    ("Conducted 30+ user interviews to identify pain points", {"cats": {"Metrics": 1, "Vague": 0}}),
    ("Assisted with research and development", {"cats": {"Metrics": 0, "Vague": 1}}),
    ("Reduced error rate by 80% through code reviews", {"cats": {"Metrics": 1, "Vague": 0}}),
    ("Contributed to quality improvements", {"cats": {"Metrics": 0, "Vague": 1}}),

    ("Led team of 6 engineers across 3 successful product releases", {"cats": {"Metrics": 1, "Vague": 0}}),
    ("Helped engineers complete tasks", {"cats": {"Metrics": 0, "Vague": 1}}),
    ("Grew Instagram followers by 12,000 in one month", {"cats": {"Metrics": 1, "Vague": 0}}),
    ("Worked on social media", {"cats": {"Metrics": 0, "Vague": 1}}),
    ("Cut processing time from 5 hours to 30 minutes", {"cats": {"Metrics": 1, "Vague": 0}}),
    ("Assisted with workflow optimization", {"cats": {"Metrics": 0, "Vague": 1}}),
    ("Increased retention by 18% in Q2", {"cats": {"Metrics": 1, "Vague": 0}}),
    ("Participated in retention strategies", {"cats": {"Metrics": 0, "Vague": 1}}),
    ("Handled 200+ customer inquiries per week", {"cats": {"Metrics": 1, "Vague": 0}}),
    ("Worked on customer support", {"cats": {"Metrics": 0, "Vague": 1}}),

    ("Launched product in 3 markets, generating $100K in revenue", {"cats": {"Metrics": 1, "Vague": 0}}),
    ("Involved in product launch efforts", {"cats": {"Metrics": 0, "Vague": 1}}),
    ("Created dashboards used by 500+ employees", {"cats": {"Metrics": 1, "Vague": 0}}),
    ("Helped create internal tools", {"cats": {"Metrics": 0, "Vague": 1}}),
    ("Wrote technical documentation for 10+ APIs", {"cats": {"Metrics": 1, "Vague": 0}}),
    ("Worked on documentation", {"cats": {"Metrics": 0, "Vague": 1}}),
    ("Improved Net Promoter Score from 42 to 65", {"cats": {"Metrics": 1, "Vague": 0}}),
    ("Helped improve customer satisfaction", {"cats": {"Metrics": 0, "Vague": 1}}),
    ("Trained 40+ employees on new systems", {"cats": {"Metrics": 1, "Vague": 0}}),
    ("Supported team training initiatives", {"cats": {"Metrics": 0, "Vague": 1}}),

    ("Managed budget of $1.2 million across 4 departments", {"cats": {"Metrics": 1, "Vague": 0}}),
    ("Managed various budget-related tasks", {"cats": {"Metrics": 0, "Vague": 1}}),
    ("Delivered 95% of projects on time and within budget", {"cats": {"Metrics": 1, "Vague": 0}}),
    ("Worked on project deadlines", {"cats": {"Metrics": 0, "Vague": 1}}),
    ("Reduced late payments by 60%", {"cats": {"Metrics": 1, "Vague": 0}}),
    ("Assisted with payment issues", {"cats": {"Metrics": 0, "Vague": 1}}),
    ("Cut server costs by 25% using spot instances", {"cats": {"Metrics": 1, "Vague": 0}}),
    ("Helped reduce server costs", {"cats": {"Metrics": 0, "Vague": 1}}),
    ("Built tool used to automate 80% of daily tasks", {"cats": {"Metrics": 1, "Vague": 0}}),
    ("Developed automation tools", {"cats": {"Metrics": 0, "Vague": 1}}),

    ("Shipped 4 new features with 0 critical bugs", {"cats": {"Metrics": 1, "Vague": 0}}),
    ("Participated in feature releases", {"cats": {"Metrics": 0, "Vague": 1}}),
    ("Monitored 12 servers and maintained 99.9% uptime", {"cats": {"Metrics": 1, "Vague": 0}}),
    ("Maintained server stability", {"cats": {"Metrics": 0, "Vague": 1}}),
    ("Increased test coverage from 65% to 95%", {"cats": {"Metrics": 1, "Vague": 0}}),
    ("Improved code quality", {"cats": {"Metrics": 0, "Vague": 1}}),
    ("Presented insights from survey of 1,200 users", {"cats": {"Metrics": 1, "Vague": 0}}),
    ("Conducted user surveys", {"cats": {"Metrics": 0, "Vague": 1}}),
    ("Reduced average call time from 10 to 6 minutes", {"cats": {"Metrics": 1, "Vague": 0}}),
    ("Improved support efficiency", {"cats": {"Metrics": 0, "Vague": 1}}),

    ("Processed 3,000+ transactions weekly with 99% accuracy", {"cats": {"Metrics": 1, "Vague": 0}}),
    ("Handled transactions and data entry", {"cats": {"Metrics": 0, "Vague": 1}}),
    ("Achieved 110% of sales target for 3 straight quarters", {"cats": {"Metrics": 1, "Vague": 0}}),
    ("Worked on sales initiatives", {"cats": {"Metrics": 0, "Vague": 1}}),
    ("Scheduled 150+ client calls each month", {"cats": {"Metrics": 1, "Vague": 0}}),
    ("Handled client communication", {"cats": {"Metrics": 0, "Vague": 1}}),
    ("Wrote 10 proposals that secured $75K in new business", {"cats": {"Metrics": 1, "Vague": 0}}),
    ("Drafted proposals for new clients", {"cats": {"Metrics": 0, "Vague": 1}}),
    ("Reduced customer wait time by 3 minutes", {"cats": {"Metrics": 1, "Vague": 0}}),
    ("Worked on customer support issues", {"cats": {"Metrics": 0, "Vague": 1}})
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

textcat.add_label("Metrics")
textcat.add_label("Non-Metric")


optimizer = nlp.begin_training()

for epoch in range(4):
    losses = {}
    batches = minibatch(TRAIN_DATA, size=2)
    for batch in batches:
        examples = []
        for text, annotations in batch:
            doc = nlp.make_doc(text)
            examples.append(Example.from_dict(doc, annotations))
        nlp.update(examples, sgd=optimizer, losses=losses)
    print(f"Epoch {epoch + 1}: Loss = {losses['textcat_multilabel']:.3f}")


nlp.to_disk("metric_classifier")
print("Model saved to 'metric_classifier'")


