# Part 1: turn each email into numbers (features) and save them
# Run this first, then run spam_detector.py

import random
import pandas as pd

random.seed(42)  # so the random changes are the same every time

money_words = ["bank", "bill", "cash", "credit", "discount", "dollars", "earn",
               "free", "invoice", "loan", "million", "money", "offer", "paid",
               "pay", "payment", "prize", "refund", "rent", "win", "won"]

# My two new features
urgency_words = ["now", "urgent", "today", "limited", "immediately", "hours",
                 "fast", "avoid", "expires", "hurry"]
action_words = ["click", "reply", "verify", "confirm", "claim", "apply", "buy", "link"]


# Count how many words in the email are in a word list
def count_words(text, word_list):
    count = 0
    for word in text.split():
        word = word.strip(".,!?:;$\"'").lower()   # "NOW!!!" becomes "now"
        if word in word_list:
            count = count + 1
    return count


# Count words in CAPITALS, like "FREE" (but not "I" or "a")
def count_caps(text):
    count = 0
    for word in text.split():
        word = word.strip(".,!?:;$\"'")
        if len(word) >= 2 and word.isupper():
            count = count + 1
    return count


# Turn one email into a row of numbers
def make_row(text, label):
    return {
        "text": text,
        "word_count": len(text.split()),
        "exclamations": text.count("!"),
        "money_words": count_words(text, money_words) + text.count("$"),
        "all_caps": count_caps(text),
        "urgency": count_words(text, urgency_words),
        "call_to_action": count_words(text, action_words),
        "label": label,
    }


# Make a near-copy of an email with one small random change
def make_variation(text):
    words = text.split()
    i = random.randint(0, len(words) - 1)
    change = random.choice(["remove", "repeat", "add !"])
    if change == "remove":
        words.pop(i)
    elif change == "repeat":
        words.insert(i, words[i])
    else:
        words[i] = words[i] + "!"
    return " ".join(words)


# The 16 emails: keep each one and make 9 variations of it

emails = pd.read_csv("data/emails.csv")
rows = []

for index, email in emails.iterrows():
    texts = [email["text"]]
    for n in range(9):
        texts.append(make_variation(email["text"]))

    for text in texts:
        row = make_row(text, email["label"])
        row["source_id"] = email["id"]   # which of the 16 emails it came from
        rows.append(row)

pd.DataFrame(rows).to_csv("data/features.csv", index=False)
print("Saved", len(rows), "emails to data/features.csv")


# Part 3.5: my three emails made to fool the model

my_emails = [
    make_row("URGENT, rent payment is past due and will be charged an extra fee "
             "of $100 if not paid by 10/5/26", "legit"),
    make_row("Hello, I'd like to get in touch about my offer earlier, it has been "
             "expanded to $350 if you would reconsider", "legit"),
    make_row("Your blackboard account has been compromised, please sign in here "
             "to change your credentials", "spam"),
]

pd.DataFrame(my_emails).to_csv("data/attack_emails.csv", index=False)
print("Saved 3 emails to data/attack_emails.csv")