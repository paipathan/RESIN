import spacy

nlp_metric = spacy.load("metric_classifier")
nlp_domain = spacy.load("domain_classifier")
nlp_actionable = spacy.load("actionable_classifier")


test_texts = [
    "Led the redesign of the onboarding process, reducing churn by 15%",
    "Participated in weekly team meetings",
    "Initiated a peer mentorship program across departments",
    "Was involved in data entry tasks",
    "Improved server response time by optimizing backend logic",
    "Helped with preparing financial reports",
    "Developed and deployed a task management tool for internal use",
    "Worked with the sales team on client outreach",
    "Designed a new logo as part of the brand refresh project",
    "Responsible for maintaining documentation"
]


def run_test(nlp):
    for text in test_texts:
        doc = nlp(text)
        cat = []
        print(f"\nText: {text}")
        print("Prediction scores:")
        for label, score in doc.cats.items():
            print(f"  {label}: {score:.4f}")
            cat.append(score)
        if(abs(cat[0]-cat[1]) == 0.4):
            predicted_label = "ehh its like in the middle bro"
        else: 
            predicted_label = max(doc.cats, key=doc.cats.get)
            print(f"Predicted class: {predicted_label}")
            
            
run_test(nlp_domain)
    
# hooooly shit im goated