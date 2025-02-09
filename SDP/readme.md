my_project/
├── app/
│   ├── app.py
│   ├── templates/
│   │    ├── login.html
│   │    ├── signup.html
│   │    └── label.html
│   └── static/
│        └── css/
│             └── style.css
├── data/
│   ├── adversarial_prompts.json   # sample adversarial prompts
│   ├── dataset.json               # generated responses go here
│   ├── labeled_dataset.json       # labels appended here
│   └── users.json                 # persistent user credentials
├── scripts/
│   ├── generate_responses.py
│   └── rlhf_train.py
└── requirements.txt