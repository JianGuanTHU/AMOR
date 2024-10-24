# data_name = "hotpotqa"
# data_name = "pubmedqa"
# data_name = "popqa"
data_name = "qasper"

use_example = True
def get_decompose_prompt(
        task,
        history,
        use_example=use_example,
        example_str=None,
        only_ipt_opt=False):
    if only_ipt_opt:
        return f"Main Question: {task}{history}"
    if use_example:
        if example_str is not None:
            decompose_prompt_example = example_str
        else:
            if data_name == "hotpotqa":
                decompose_prompt_example = """HERE ARE SEVERAL EXAMPLES:
====Examples Start====
(1) Main Question: What U.S Highway gives access to Zilpo Road, and is also known as Midland Trail?
Output: [Next] How can Zilpo Road be accessed?

(2) Main Question: Which magazine was started first Arthur's Magazine or First for Women?
Solved Sub-Questions:
1. Q: When was Arthur's Magazine started? A: 1844-1846
Output: [Next] When was First for Women magazine started?

(3) Main Question: Which magazine was started first Arthur's Magazine or First for Women?
Solved Sub-Questions:
1. Q: When was Arthur's Magazine started? A: 1844-1846
2. Q: When was First for Women magazine started? A: 1989
Output: [Finish]
====Examples End====\n\n"""
            elif data_name == "pubmedqa":
                decompose_prompt_example = """HERE ARE SEVERAL EXAMPLES:
====Examples Start====
(1) Main Question: Does histologic chorioamnionitis correspond to clinical chorioamnionitis?
Output: [Next] Does histologic chorioamnionitis correspond to clinical chorioamnionitis?

(2) Main Question: Can vitamin C prevent complex regional pain syndrome in patients with wrist fractures?
Solved Sub-Questions:
1. Q: Can vitamin C prevent complex regional pain syndrome in patients with wrist fractures? A: yes
Output: [Finish]
====Examples End====\n\n"""
            elif data_name == "popqa":
                decompose_prompt_example = """HERE ARE SEVERAL EXAMPLES:
====Examples Start====
(1) Main Question: What is the seed lexicon?
Output: [Next] What is the seed lexicon?

(2) Main Question: Who is the author of Homeland?
Solved Sub-Questions:
1. Q: Who is the author of Homeland? A: Cory Doctorow
Output: [Finish]
====Examples End====\n\n"""
            elif data_name == "qasper":
                decompose_prompt_example = """HERE ARE SEVERAL EXAMPLES:
====Examples Start====
(1) Main Question: In what city was Runa Akiyama born?
Output: [Next] In what city was Runa Akiyama born?

(2) Main Question: Who is the author of Homeland?
Solved Sub-Questions:
1. Q: How much labeled data is available for these two languages? A: 1,101 sentences (26k tokens)
Output: [Finish]
====Examples End====\n\n"""
    else:
        decompose_prompt_example = ""
    decompose_prompt = f"""Please continue to decompose the provided main question into answerable sub-questions following previously already solved sub-questions. There are two cases as follows:
(1) [Next] If the question requires further decomposition: Identify and output the next logical sub-question that must be addressed in order to progress towards answering the main question.
(2) [Finish] It means the question does not require further decomposition and can be answered as is.

{decompose_prompt_example}Now Please Complete the Following Task. Please ensure that each sub-question is specific enough to understand in isolation.
Main Question: {task}{history}
Output:"""
    return decompose_prompt


def get_relevance_prompt(
        task,
        history,
        query,
        docs,
        use_example=use_example, 
        example_str=None):
    if use_example:
        if example_str is not None:
            relevance_prompt_example = example_str
        else:
            if data_name == "hotpotqa":
                relevance_prompt_example = """HERE ARE SEVERAL EXAMPLES:
====Examples Start====
(1) Main Question: Which magazine was started first Arthur's Magazine or First for Women?
Next Sub-Question: When was Arthur's Magazine started?
Documents: (title: Arthur's Magazine) Arthur's Magazine Arthur's Magazine (1844-1846) was an American literary periodical published in Philadelphia in the 19th century. Edited by Timothy Shay Arthur, it featured work by Edgar A. Poe, J.H. Ingraham, Sarah Josepha Hale, Thomas G. Spear, and others. In May 1846 it was merged into Godey's Lady's Book. A few years later Arthur would launch a new publication entitled Arthur's Home Magazine.
Next Sub-Question: When was Arthur's Magazine started?
Output: [Relevant]

(2) Main Question: Which magazine was started first Arthur's Magazine or First for Women?
Solved Sub-Questions:
1. Q: When was Arthur's Magazine started? A: 1844-1846
Next Sub-Question: When was First for Women magazine started?
Documents: (title: History of women's magazines) History of women's magazines This article addresses the history of women's magazines. In 1693 the first issue of the first women's magazine in Britain, "The Ladies' Mercury", was published. In 1857 the first women's magazine in Gujarati, "Streebodh", was established by Parsi social activists. In 1892 the first women's magazine in Egypt, and indeed in all the Arab countries, "Al Fatat", was established by Hind Nawfal. In the period before the American Civil War, "Godey's Lady's Book" was a United States women's magazine that was the most widely circulated magazine. Its circulation rose from 70,000 in the 1840s to 150,000
Next Sub-Question: When was First for Women magazine started?
Output: [Irrelevant]

(3) Main Question: What U.S Highway gives access to Zilpo Road, and is also known as Midland Trail?
Next Sub-Question: How can Zilpo Road be accessed?
Documents: (title: Zilpo Road) constructed on the Licking River by the Army Corps of Engineers. The Zilpo Recreation Area is a park offering wooded campsites and a variety of facilities. On the other side of the lake is the Twin Knobs Recreation Area. which has additional campsites. Zilpo Road Zilpo Road is a National Forest Scenic Byway in the forested hills of eastern Kentucky in the United States. The byway starts south of Salt Lick and can be accessed by Kentucky Route 211 (KY 2112). The byway travels through the Daniel Boone National Forest and ends on the western shore of Cave Run Lake
Next Sub-Question: How can Zilpo Road be accessed?
Output: [Relevant]
====Examples End====\n\n"""
            elif data_name == "pubmedqa":
                relevance_prompt_example = """HERE ARE SEVERAL EXAMPLES:
====Examples Start====
(1) Main Question: Does histologic chorioamnionitis correspond to clinical chorioamnionitis?
Next Sub-Question: Does histologic chorioamnionitis correspond to clinical chorioamnionitis?
Documents: (title: 18251357) OBJECTIVE: To evaluate the degree to which histologic chorioamnionitis, a frequent finding in placentas submitted for histopathologic evaluation, correlates with clinical indicators of infection in the mother.
Next Sub-Question: Does histologic chorioamnionitis correspond to clinical chorioamnionitis?
Output: [Relevant]

(2) Main Question: Can vitamin C prevent complex regional pain syndrome in patients with wrist fractures?
Next Sub-Question: Can vitamin C prevent complex regional pain syndrome in patients with wrist fractures?
Documents: (title: 19504993) METHODS: The medical records of 14 patients with Fournier's gangrene who presented at the University Hospital Center \"Mother Teresa\" from January 1997 to December 2006 were reviewed retrospectively to analyze the outcome and identify the risk factor and prognostic indicators of mortality.\nRESULTS: Of the 14 patients, 5 died and 9 survived. Mean age was 54 years (range from 41-61): it was 53 years in the group of survivors and 62 years in deceased group. There was a significant difference in leukocyte count between patients who survived (range 4900-17000/mm) and those died (range 20.300-31000/mm3). Mean hospital stay was about 19 days (range 2-57 days).
Next Sub-Question: Can vitamin C prevent complex regional pain syndrome in patients with wrist fractures?
Output: [Irrelevant]
====Examples End====\n\n"""
            elif data_name == "popqa":
                relevance_prompt_example = """HERE ARE SEVERAL EXAMPLES:
====Examples Start====
(1) Main Question: Who is the author of Homeland?
Next Sub-Question: Who is the author of Homeland?
Documents: (title: Homeland (Doctorow novel)) Homeland (Doctorow novel) Homeland is a novel by Cory Doctorow, published by Tor Books. It is a sequel to Doctorow's earlier novel, \"Little Brother\". It was released in hardback on February 5, 2013 and subsequently released for download under a Creative Commons license on Doctorow's website two weeks later on February 19, 2013. The novel includes two afterword essays by computer security researcher and hacker Jacob Appelbaum, and computer programmer and Internet activist Aaron Swartz. Homeland is dedicated to Doctorow's wife and daughter, Alice and Poesy. As in \"Little Brother\", Doctorow also dedicates each e-book chapter of Homeland to a
Next Sub-Question: Who is the author of Homeland?
Output: [Relevant]

(2) Main Question: In what city was Runa Akiyama born?
Next Sub-Question: In what city was Runa Akiyama born?
Documents: (title: Angela Aki) Angela Aki Aki was born in the small town of Itano, population 14,600, in the mostly rural island of Shikoku. Aki began to take piano lessons when she was three years old and lived in Tokushima through sixth grade and spent her junior high school days in Okayama. She has admitted that growing up in rural Japan proved very difficult, as she was bullied and she turned to the piano as an escape from the isolation she felt. She grew up listening to a mix of enka, The Carpenters and The Bee Gees. Aki moved to Hawaii when she was
Next Sub-Question: In what city was Runa Akiyama born?
Output: [Irrelevant]
====Examples End====\n\n"""
            elif data_name == "qasper":
                relevance_prompt_example = """HERE ARE SEVERAL EXAMPLES:
====Examples Start====
(1) Main Question: What is the seed lexicon?
Next Sub-Question: What is the seed lexicon?
Documents: (title: Proposed Method ::: Discourse Relation-Based Event Pairs) Proposed Method ::: Discourse Relation-Based Event Pairs Our method requires a very small seed lexicon and a large raw corpus. We assume that we can automatically extract discourse-tagged event pairs, $(x_i1, x_i2)$ ($i=1, \\cdots $) from the raw corpus. We refer to $x_i1$ and $x_i2$ as former and latter events, respectively. As shown in Figure FIGREF1, we limit our scope to two discourse relations: Cause and Concession.
Next Sub-Question: What is the seed lexicon?
Output: [Relevant]

(2) Main Question: How much labeled data is available for these two languages?
Next Sub-Question: How much labeled data is available for these two languages?
Documents: (title: Conclusion) Conclusion In this study, we analysed distant supervision techniques and label-noise handling for NER in Hausa and Yor\u00f9b\u00e1, two languages from developing countries. We showed that they can be successfully leveraged in a realistic low-resource scenario to double a classifier's performance. 
Next Sub-Question: How much labeled data is available for these two languages?
Output: [Irrelevant]
====Examples End====\n\n"""
    else:
        relevance_prompt_example = ""
    relevance_prompt = f"""Given a sub-question derived from the main question and a document with its title (an entity name), please assess whether the title/entity is relevant with the sub-question based on the title and shown content of the document. Assign one of the following two categories:
(1) [Relevant]: Choose this category if the given document is relevant with the sub-question.
(2) [Irrelevant]: Select this category if the document is irrelevant with the sub-question.

{relevance_prompt_example}Now Please Complete the Following Task:
Main Question: {task}{history}
Next Sub-Question: {query}
Documents: {docs}
Next Sub-Question: {query}
Output:"""
    return relevance_prompt


def get_solve_prompt(
        task,
        history,
        query,
        docs,
        use_example=use_example, example_str=None):
    if use_example:
        if example_str is not None:
            solve_prompt_example = example_str
        else:
            if data_name == "hotpotqa":
                solve_prompt_example = """HERE ARE SEVERAL EXAMPLES:
====Examples Start====
(1) Main Question: Which magazine was started first Arthur's Magazine or First for Women?
Solved Sub-Questions:
1. Q: When was First for Women magazine started? A: 1989
Next Sub-Question: When was Arthur's Magazine started?
Passages: [1] (title: Arthur's Magazine) He was also the author of dozens of stories for Godey's Lady's Book, the most popular American monthly magazine in the antebellum era, and he published and edited his own Arthur's Home Magazine, a periodical in the Godey's model
[2] (title: Arthur's Magazine) Arthur's Magazine Arthur's Magazine (1844-1846) was an American literary periodical published in Philadelphia in the 19th century. Edited by Timothy Shay Arthur, it featured work by Edgar A. Poe, J.H. Ingraham, Sarah Josepha Hale, Thomas G. Spear, and others. In May 1846 it was merged into Godey's Lady's Book. A few years later Arthur would launch a new publication entitled Arthur's Home Magazine.
[3] (title: Arthur's Magazine) The articles were widely reprinted and helped fuel the establishment of Washingtonian groups across the country. Arthur’s newspaper sketches were collected in book form as Six Nights with the Washingtonians (1842). Six Nights went through many editions and helped establish Arthur in the public eye as an author associated with the temperance movement.
Next Sub-Question: When was Arthur's Magazine started?
Output: [Answerable] Answer: 1844-1846; Relevant Passage ID: [2]

(2) Main Question: What U.S Highway gives access to Zilpo Road, and is also known as Midland Trail?
Next Sub-Question: How can Zilpo Road be accessed?
Passages: [1] (title: Zilpo Road) the city which transports people in and out of the city. Local bus terminals are also present and are primarily used for transportation in neighboring towns and inside the Subic Bay Freeport Zone. Olongapo is accessible through the National Highway (via Zigzag Road) from Hermosa and Dinalupihan, Bataan. The National Highway cuts through the city center and goes through north up to Barangay Barreto and then on to the neighboring town of Subic, and the rest of the towns in Zambales up to Pangasinan province. Another access to the city is via SCTEX and Subic–Tipo Expressway exiting to the gates
[2] (title: Zilpo Road) Grand Terrace. Access provides public transportation services for persons who are physically or cognitively unable to use regular bus service (ADA certified and/or Omnitrans Disability Identification Card holders). Access operates curb to- curb service with minibuses or vans, complementing the Omnitrans fixed-route bus system. The Access service area is defined as up to 3/4 mile on either side of an existing fixed route. Service is available on the same days and at the same times that fixed-route services operate. Omnilink is a general-public, demand-response service that operates in Yucaipa and Chino Hills. This service circulates through a defined, low-density service-area
Next Sub-Question: How can Zilpo Road be accessed?
Output: [Unanswerable]
====Examples End====\n\n"""
            elif data_name == "pubmedqa":
                solve_prompt_example = """HERE ARE SEVERAL EXAMPLES:
====Examples Start====
(1) Main Question: Does histologic chorioamnionitis correspond to clinical chorioamnionitis?
Next Sub-Question: Does histologic chorioamnionitis correspond to clinical chorioamnionitis?
Passages: [1] (title: 18251357) OBJECTIVE: To evaluate the degree to which histologic chorioamnionitis, a frequent finding in placentas submitted for histopathologic evaluation, correlates with clinical indicators of infection in the mother.\nSTUDY DESIGN: A retrospective review was performed on 52 cases with a histologic diagnosis of acute chorioamnionitis from 2,051 deliveries at University Hospital, Newark, from January 2003 to July 2003. Third-trimester placentas without histologic chorioamnionitis (n = 52) served as controls. Cases and controls were selected sequentially. Maternal medical records were reviewed for indicators of maternal infection.\nRESULTS: Histologic chorioamnionitis was significantly associated with the usage of antibiotics (p = 0.0095) and a higher mean white blood cell count (p = 0.018). The presence of 1 or more clinical indicators was significantly associated with the presence of histologic chorioamnionitis (p = 0.019).
Next Sub-Question: Does histologic chorioamnionitis correspond to clinical chorioamnionitis?
Output: [Answerable] Answer: yes; Relevant Passage ID: [1]

(2) Main Question: Can the condition of the cell microenvironment of mediastinal lymph nodes help predict the risk of metastases in non-small cell lung cancer?
Next Sub-Question: Can the condition of the cell microenvironment of mediastinal lymph nodes help predict the risk of metastases in non-small cell lung cancer?
Passages: [1] (title: 11799314) BACKGROUND: Gallbladder carcinoma is characterized by delayed diagnosis, ineffective treatment and poor prognosis. Surgical resection has been thought to be the treatment of choice, while the role of radiotherapy as adjuvant or palliative treatment has not been fully clarified in the literature.\nPATIENT AND METHODS: We present the case of a 45-year-old female, with unresectable gallbladder carcinoma, grade IV, histologically diagnosed during laparotomy. The patient was treated with palliative intent with percutaneous transhepatic biliary drainage. Furthermore, she received external radiotherapy by (60)Co, using a three-field technique (anterior-posterior and right lateral). The total dose was 3,000 cGy in 10 fractions, with 300 cGy per fraction, 5 days weekly.\nRESULTS: The patient showed clinico-laboratory improvement and was discharged with a permanent percutaneous transhepatic endoprosthesis. During follow-up (10 and 12 months postirradiation), abdominal CTs showed no local extension of the tumor, while the patient had a good performance status. So far, 1 year after the diagnosis of gallbladder cancer she is still alive.
Next Sub-Question: Can the condition of the cell microenvironment of mediastinal lymph nodes help predict the risk of metastases in non-small cell lung cancer?
Output: [Unanswerable]
====Examples End====\n\n"""
            elif data_name == "popqa":
                solve_prompt_example = """HERE ARE SEVERAL EXAMPLES:
====Examples Start====
(1) Main Question: Who is the author of Homeland?
Next Sub-Question: Who is the author of Homeland?
Passages: [1] (title: Homeland (Doctorow novel)) Homeland is dedicated to Doctorow's wife and daughter, Alice and Poesy.[2] As in Little Brother, Doctorow also dedicates each e-book chapter of Homeland to a different bookstore: Chapters/Indigo, BakkaPhoenix Books, Barnes & Noble, Wild Rumpus, University Book Store at the University of Washington
[2] (title: Homeland (Doctorow novel)) Homeland (Doctorow novel) Homeland is a novel by Cory Doctorow, published by Tor Books. It is a sequel to Doctorow's earlier novel, \"Little Brother\". It was released in hardback on February 5, 2013 and subsequently released for download under a Creative Commons license on Doctorow's website two weeks later on February 19, 2013. 
[3] (title: Homeland (Doctorow novel)) The novel includes two afterword essays by computer security researcher and hacker Jacob Appelbaum, and computer programmer and Internet activist Aaron Swartz. Homeland is dedicated to Doctorow's wife and daughter, Alice and Poesy. As in \"Little Brother\", Doctorow also dedicates each e-book chapter of Homeland to a
Next Sub-Question: Who is the author of Homeland?
Output: [Answerable] Answer: Cory Doctorow; Relevant Passage ID: [2]

(2) Main Question: In what city was Runa Akiyama born?
Next Sub-Question: In what city was Runa Akiyama born?
Passages: [1] (title: Angela Aki) Angela Aki Aki was born in the small town of Itano, population 14,600, in the mostly rural island of Shikoku. Aki began to take piano lessons when she was three years old and lived in Tokushima through sixth grade and spent her junior high school days in Okayama. 
[2] (title: Angela Aki) She has admitted that growing up in rural Japan proved very difficult, as she was bullied and she turned to the piano as an escape from the isolation she felt. She grew up listening to a mix of enka, The Carpenters and The Bee Gees. Aki moved to Hawaii when she was
[3] (title: Angela Aki) the writer under the supervision of R.A. Salvatore. The adaption of \"Homeland\", volume 1, received a positive review from George Galuschak of \"Kliatt\" magazine, who said, \"I am not a big sword & sorcery buff, but I enjoyed this graphic novel.
Next Sub-Question: In what city was Runa Akiyama born?
Output: [Unanswerable]
====Examples End====\n\n"""
            elif data_name == "qasper":
                solve_prompt_example = """HERE ARE SEVERAL EXAMPLES:
====Examples Start====
(1) Main Question: What is the seed lexicon?
Next Sub-Question: What is the seed lexicon?
Passages: [1] (title: Proposed Method ::: Discourse Relation-Based Event Pairs) Proposed Method ::: Discourse Relation-Based Event Pairs Our method requires a very small seed lexicon and a large raw corpus. We assume that we can automatically extract discourse-tagged event pairs, $(x_i1, x_i2)$ ($i=1, \\cdots $) from the raw corpus. We refer to $x_i1$ and $x_i2$ as former and latter events, respectively. As shown in Figure FIGREF1, we limit our scope to two discourse relations: Cause and Concession.
[2] (title: Proposed Method ::: Discourse Relation-Based Event Pairs) The seed lexicon consists of positive and negative predicates. If the predicate of an extracted event is in the seed lexicon and does not involve complex phenomena like negation, we assign the corresponding polarity score ($+1$ for positive events and $-1$ for negative events) to the event. We expect the model to automatically learn complex phenomena through label propagation. Based on the availability of scores and the types of discourse relations, we classify the extracted event pairs into the following three types.
[3] (title: Proposed Method ::: Discourse Relation-Based Event Pairs) The models in the top block performed considerably better than the random baselines. The performance gaps with their (semi-)supervised counterparts, shown in the middle block, were less than 7%. This demonstrates the effectiveness of discourse relation-based label propagation.\nComparing the model variants, we obtained the highest score with the BiGRU encoder trained with the AL+CA+CO dataset. BERT was competitive but its performance went down if CA and CO were used in addition to AL. We conjecture that BERT was more sensitive to noises found more frequently in CA and CO.
Next Sub-Question: What is the seed lexicon?
Output: [Answerable] Answer: seed lexicon consists of positive and negative predicates; Relevant Passage ID: [2]

(2) Main Question: How much labeled data is available for these two languages?
Next Sub-Question: How much labeled data is available for these two languages?
Passages: [1] (title: Conclusion) Conclusion In this study, we analysed distant supervision techniques and label-noise handling for NER in Hausa and Yor\u00f9b\u00e1, two languages from developing countries. We showed that they can be successfully leveraged in a realistic low-resource scenario to double a classifier's performance. 
[2] (title: Conclusion) If model size is not a constraint, the more complex BERT model clearly outperforms the smaller Bi-LSTM architecture. Nevertheless, there is still a large gap between the best performing model on Yor\u00f9b\u00e1 with 66 F1-score and the state-of-the-art in English around 90.
[3] (title: Conclusion) We see several interesting follow-ups to these evaluations. In the future, we want to evaluate if noise handling methods can also allow the more complex BERT model to benefit from distant supervision. Regarding the model complexity, it would be interesting to experiment with more compact models like DistilBERT BIBREF42 that reach a similar performance with a smaller model size for high-resource settings.
Next Sub-Question: How much labeled data is available for these two languages?
Output: [Unanswerable]
====Examples End====\n\n"""
    else:
        solve_prompt_example = ""
    solve_prompt = f"""Please assess whether the sub-question derived from the main question can be answered using the information from the provided passages. Your evaluation should categorize the sufficiency of the information in the passages with respect to the sub-question. Assign one of the following three categories:
(1) [Unanswerable]: Choose this category if the given passages do not contain information to answer it directly.
(2) [Answerable]: Use this category if one of the given passages contains sufficient information to directly answer the sub-question. Provide a clear and concise answer to the sub-question, and the ID of the the corresponding passage.

{solve_prompt_example}Now Please Complete the Following Task:
Main Question: {task}{history}
Next Sub-Question: {query}
Passages: {docs}
Next Sub-Question: {query}
Output:"""
    return solve_prompt

def get_finish_prompt(
        task,
        psgs,
        history,
        use_example=use_example, example_str=None):
    if use_example:
        if example_str is not None:
            finish_prompt_example = example_str
        else:
            # without thought
            if data_name == "hotpotqa":
                finish_prompt_example = """HERE ARE SEVERAL EXAMPLES:
====Examples Start====
(1) Question: Which magazine was started first Arthur's Magazine or First for Women?
Passags: [1] (title: Arthur's Magazine) Arthur's Magazine Arthur's Magazine (1844-1846) was an American literary periodical published in Philadelphia in the 19th century. Edited by Timothy Shay Arthur, it featured work by Edgar A. Poe, J.H. Ingraham, Sarah Josepha Hale, Thomas G. Spear, and others. In May 1846 it was merged into Godey's Lady's Book. A few years later Arthur would launch a new publication entitled Arthur's Home Magazine.
[2] (title: First for Women) First for Women is a woman's magazine published by A360media in the US. The magazine was started in 1989 by Bauer Media Group. In 2011 the circulation of the magazine was 1,310,696 copies.
Question: Which magazine was started first Arthur's Magazine or First for Women?
Output: Arthur's Magazine

(2) Question: What U.S Highway gives access to Zilpo Road, and is also known as Midland Trail?
Passages: [1] (title: Zilpo Road) Zilpo Road Zilpo Road is a National Forest Scenic Byway in the forested hills of eastern Kentucky in the United States. The byway starts south of Salt Lick and can be accessed by Kentucky Route 211 (KY 2112). The byway travels through the Daniel Boone National Forest and ends on the western shore of Cave Run Lake at the Zilpo Recreation Area. It follows FSR 918, which is a two-lane paved road suitable for all motor vehicles and is usually open throughout the year. Cave Run Lake is one of the main attractions of this byway. It is a lake
[2] (title: Morehead, Kentucky) Morehead is a home rule-class city[5] located along US 60 (the historic Midland Trail) and Interstate 64 in Rowan County, Kentucky, in the United States. It is the seat of its county.[6] The population was 7,151 at the time of the 2020 U.S. census.[7]
Question: What U.S Highway gives access to Zilpo Road, and is also known as Midland Trail?
Output: US 60
====Examples End====\n\n"""
            elif data_name == "pubmedqa":
                finish_prompt_example = """HERE ARE SEVERAL EXAMPLES:
====Examples Start====
(1) Question: Does histologic chorioamnionitis correspond to clinical chorioamnionitis?
Passags: [1] (title: 18251357) OBJECTIVE: To evaluate the degree to which histologic chorioamnionitis, a frequent finding in placentas submitted for histopathologic evaluation, correlates with clinical indicators of infection in the mother.\nSTUDY DESIGN: A retrospective review was performed on 52 cases with a histologic diagnosis of acute chorioamnionitis from 2,051 deliveries at University Hospital, Newark, from January 2003 to July 2003. Third-trimester placentas without histologic chorioamnionitis (n = 52) served as controls. Cases and controls were selected sequentially. Maternal medical records were reviewed for indicators of maternal infection.\nRESULTS: Histologic chorioamnionitis was significantly associated with the usage of antibiotics (p = 0.0095) and a higher mean white blood cell count (p = 0.018). The presence of 1 or more clinical indicators was significantly associated with the presence of histologic chorioamnionitis (p = 0.019).
Question: Does histologic chorioamnionitis correspond to clinical chorioamnionitis?
Output: yes

(2) Question: Is a mandatory general surgery rotation necessary in the surgical clerkship?
Passages: [1] (title: 9645785) BACKGROUND: Changes in the spectrum of general surgery and the delivery of surgical care have placed the requirement for a mandatory general surgery rotation in the surgical clerkship in question.\nMETHODS: We tested the hypothesis that equal mastery of surgical clerkship objectives can be obtained in a clerkship with and without general surgery. Students chose any two surgical rotations and were assessed by written examination, objective structured clinical examination (OSCE), ward evaluations, self-assessment objectives questionnaire, and satisfaction survey.\nRESULTS: Data for 54 students showed no differences in scores between groups on any parameter. No specific concerns related to the absence of general surgery were identified.
Question: Is a mandatory general surgery rotation necessary in the surgical clerkship?
Output: no
====Examples End====\n\n"""
            elif data_name == "popqa":
                finish_prompt_example = """HERE ARE SEVERAL EXAMPLES:
====Examples Start====
(1) Question: In what city was Runa Akiyama born?
Passags: [1] (title: Runa Akiyama) Runa Akiyama (あきやまるな, Akiyama Runa, April 17, 1954 – March 8, 2014[1]) was a Japanese voice actress. Her real name was Teruko Akiyama (秋山照子, Akiyama Teruko). She was born in the Metropolitan area of Tokyo, Japan. She was affiliated with the voice talent management group 81 Produce at the time of her death.
Question: In what city was Runa Akiyama born?
Output: Tokyo

(2) Question: Who is the author of Homeland?
Passages: [1] (title: Homeland (Doctorow novel)) Homeland (Doctorow novel) Homeland is a novel by Cory Doctorow, published by Tor Books. It is a sequel to Doctorow's earlier novel, \"Little Brother\". It was released in hardback on February 5, 2013 and subsequently released for download under a Creative Commons license on Doctorow's website two weeks later on February 19, 2013. The novel includes two afterword essays by computer security researcher and hacker Jacob Appelbaum, and computer programmer and Internet activist Aaron Swartz. Homeland is dedicated to Doctorow's wife and daughter, Alice and Poesy. As in \"Little Brother\", Doctorow also dedicates each e-book chapter of Homeland to a
Question: Who is the author of Homeland?
Output: Cory Doctorow
====Examples End====\n\n"""
            elif data_name == "qasper":
                finish_prompt_example = """HERE ARE SEVERAL EXAMPLES:
====Examples Start====
(1) Question: What is the seed lexicon?
Passags: [1] (title: Proposed Method ::: Discourse Relation-Based Event Pairs) The seed lexicon consists of positive and negative predicates. If the predicate of an extracted event is in the seed lexicon and does not involve complex phenomena like negation, we assign the corresponding polarity score ($+1$ for positive events and $-1$ for negative events) to the event. We expect the model to automatically learn complex phenomena through label propagation. Based on the availability of scores and the types of discourse relations, we classify the extracted event pairs into the following three types.
Question: What is the seed lexicon?
Output: seed lexicon consists of positive and negative predicates

(2) Question: How much labeled data is available for these two languages?
Passages: [1] (title: Background & Methods ::: Datasets & Embeddings) The NER data used in this work is the annotated corpus of Global Voices news articles recently released by BIBREF22. The dataset consists of 1,101 sentences (26k tokens) divided into 709 training sentences, 113 validation sentences and 279 test sentences based on 65%/10%/25% split ratio. The named entities in the dataset are personal names (PER), organization (ORG), location (LOC) and date & time (DATE). All other tokens are assigned a tag of "O".
Question: How much labeled data is available for these two languages?
Output: 1,101 sentences (26k tokens)
====Examples End====\n\n"""
    else:
        finish_prompt_example = ""
    finish_prompt = (f"""Answer the question based on the provided passages and decomposed question-answering pairs. Your output should be 'yes/no' or a short entity.

{finish_prompt_example}Question: {task}
Passages: {psgs}{history}
Question: {task}
Output:""")
    return finish_prompt
