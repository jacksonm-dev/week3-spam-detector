# Week 3 – Spam Detector: Analysis Report

## Part 1 — Data and features

The dataset starts from 16 labelled emails (8 spam, 8 legitimate). prepare_data.py keeps each original and makes 9 slightly varied copies of it (removing a word, repeating a word or adding a !), giving 160 rows. Each email is turned into numbers, and the model never sees the text itself.

Given features: word_count, exclamations, money_words, all_caps.

### My two features

- urgency (count of words like *now, urgent, today, limited, fast*): this helps spot spam because scammers don't want you to think, they want you to hurry and give them your money, whereas real emails rarely need an instant reply.
- call_to_action (count of words like *click, reply, verify, confirm, claim, buy*): this helps spot spam because spam emails often require you to click a link, send money to an address or buy something, whereas legitimate emails mostly just tell you something or chat.

## Part 2 — Training

The 160 rows are split at random into a training set (120 rows, 75%) and a test set (40 rows, 25%), keeping the spam/legit balance the same in both. A logistic regression model is trained on the training set only.

### Why the split exists

The split exists to tell apart the score on training data from the score on unseen material. You can study for an exam and get 100% of the practice answers right, but the real exam will often be re-worded to see whether you can apply what you studied to new questions. If we tested the model on the same emails it trained on, it would only be recalling answers it has already seen, and the score would look better than it deserves. What we actually care about is how it handles new emails arriving in someone's inbox, so it has to be tested on emails it has never seen.

### Accuracy

Reported accuracy: 85% (on the test set).

## Part 3 — Evaluation

### 3.1 Confusion matrix (test set, 40 emails)

| | Predicted spam | Predicted legit |
|---|---|---|
| Actually spam | 17 (spam correctly caught) | 3 (spam wrongly let through) |
| Actually legit | 3 (legit wrongly marked spam) | 17 (legit correctly passed) |

All 6 mistakes come from just two original emails:
- Email 5 (spam, the "prince" scam) was let through 3 times.
- Email 15 (legit, the rent-share message) was marked as spam 3 times.

Why these two? The prince email (email 5) gets through because it has a high word count, no exclamation marks, no capitals, and no urgency or call-to-action words, which is not what the model expects a scam to look like, even though it has a lot of money words. The rent-share email (email 15) gets wrongly flagged because it has 6 money words, the highest of any email in the dataset. Seen only as numbers, the two emails look almost the same: long, calm and full of money talk. The model has no way to tell a polite scam from a flatmate asking for rent.

### 3.2 Precision and recall

| | Calculation | Score |
|---|---|---|
| Precision | 17 ÷ (17 + 3) | 85% |
| Recall | 17 ÷ (17 + 3) | 85% |

Precision is about the spam folder: when the filter says an email is spam, how often is it right? Our filter is right 85% of the time, so about 3 in every 20 emails in the spam folder are real emails that shouldn't be there. Low precision means real emails, like a party invitation or a rent reminder, keep getting lost in the spam folder.

Recall is about all the spam that was sent: how much of it did the filter catch? Our filter caught 85%, so about 3 in every 20 spam emails still reach the inbox. Low recall means your inbox fills up with spam that got past the filter.

The two scores are equal here only because the model made 3 mistakes of each kind.

### 3.3 Which error is worse?

I think precision matters more for this filter. Important emails like job offers cannot be missed or misplaced, because nobody checks their spam folder and the opportunity is simply lost. Spam that gets through (low recall) is just an annoyance and is easy to deal with by deleting it. So I would rather my filter let a bit of spam through than hide a real email.

What I would change: make the filter stricter about calling something spam. The model gives each email a spam probability between 0 and 1, and by default anything above 0.5 counts as spam. Raising that bar (for example to 0.7) means it only marks an email as spam when it is very sure, so fewer real emails end up in the spam folder. The cost is that more spam gets through to the inbox, and that is the trade-off I am choosing to accept.

### 3.4 What did the model learn?

| Feature | Weight |
|---|---|
| call_to_action | 2.80 |
| urgency | 1.15 |
| money_words | 0.62 |
| all_caps | 0.35 |
| exclamations | 0.24 |
| word_count | −0.05 |

A positive weight pushes an email towards spam, and a negative one towards legit. call_to_action carried the most influence. My guess was urgency, which came second, so both of my new features beat all four of the given ones.

It makes sense that call_to_action has such a high weight, because in our data none of the legit emails contained a call-to-action word and every one of the spam emails did.

I wasn't surprised that exclamation marks and capitals ranked low. Spam does use them more on average, but they are spread across both kinds of email: the legit birthday email has 5 exclamation marks and the NASA email has 3 capitalised words, while some spam (the prince and parcel emails) has none. So they are not a reliable sign on their own.

### 3.5 Trying to break it

I wrote three emails designed to fool the classifier and ran them through it.

"URGENT, rent payment is past due and will be charged an extra fee of $100 if not paid by 10/5/26", model returns legit while it was actually spam. **Fooled it**
"Hello, I'd like to get in touch about my offer earlier, it has been expanded to $350 if you would reconsider", model returns legit while it was actually legit. **Not fooled**
"Your blackboard account has been compromised, please sign in here to change your credentials", model returns spam while it was actually legit.**Fooled it**

2 out of 3 fooled the model.

This shows the model hasn't learned what spam is. It has learned which words appeared in our 16 examples.

### 3.6 What is wrong with the data?

The 160 rows are really only 16 emails, each copied 10 times with one tiny change (a word removed, a word repeated or a ! added). When the rows were split at random, 15 of the 16 original emails had copies in both the training set and the test set. The model had effectively seen its own exam paper. For example, the test set contained "Hi, I am a prince prince with 10 million dollars…", which is almost identical to a training email.

I don't trust the 85% as much any more. The variations are so small that they barely change the six numbers, so a test copy looks almost the same to the model as the training copy it came from. The test is mostly checking whether the model remembers emails it has already seen, not whether it can handle genuinely new ones. On truly new emails I would expect the score to be lower, and with only 16 real emails one tricky email (like the prince) could change the result a lot.

To get a more honest score I would:
- Use many more than 16 real, different emails, so the model learns general patterns instead of memorising a few examples.
- Split by original email, not by row, so no near-copy of a test email is ever in the training set.

Until then, the 85% should be treated as an overestimate of how well the filter would work on real, new emails.