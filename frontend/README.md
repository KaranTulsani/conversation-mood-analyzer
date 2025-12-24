1. Simple sanity check (clear emotions)
Hey, how are you?
I am feeling really happy today!
That is great to hear.
Yes, everything is going well.


Expected:

Mostly positive

First line maybe neutral

🔴 2. Clear negative conversation
I am really tired of this job.
Nothing seems to work out.
My manager keeps shouting at me.
I feel completely drained.


Expected:

Mostly negative

🟣 3. Mixed emotions (realistic)
Hey, how was your day?
It started off okay.
Then work got very stressful.
But I managed to finish everything.
Feeling relieved now.


Expected:

Neutral → Negative → Positive shift
This tests context change, which your LSTM is meant for.

🟡 4. Subtle / ambiguous emotions (harder)
How are things going?
Same as usual.
Nothing special really.
Just another day.


Expected:

Mostly neutral

This checks over-prediction problems.

🔥 5. Stress → hope (important test)
I have been under a lot of pressure lately.
Deadlines are killing me.
I barely get any rest.
But I believe things will improve soon.


Expected:

Negative → Slightly positive at the end
If your model captures this, that’s a BIG win.

😈 6. Sarcasm (model will struggle — that’s OK)
Oh wow, another meeting.
Just what I needed today.
This is amazing.
Absolutely love it.


Expected:

Model may say positive (this is NORMAL)

Sarcasm is hard even for big models

👉 You can mention this limitation in interviews (actually impressive).

❤️ 7. Emotional support conversation
I feel really low today.
I am sorry you are feeling that way.
Talking to you helps a lot.
Thank you for being there.


Expected:

Negative → Positive shift

🧠 8. Interview-ready demo conversation (BEST ONE)

Use this when showing the project:

Hey, how are you?
I am feeling exhausted lately.
Work has been really stressful.
But I am trying to stay positive.
I know things will get better.


Why this is perfect:

Shows context

Shows emotional transition

Makes your LSTM choice look justified