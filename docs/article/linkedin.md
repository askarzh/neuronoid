# I nodded along about AI for months. A 1988 game showed me how little I understood

![The final screen of DOS Arkanoid: the VAUS ship escapes after the DOH boss is destroyed](img/final.png)

For a long time I did what I suspect many people in leadership roles do. Agents, vibe coding, Software 3.0: every
month a new term, in every meeting, in every newsletter. I nodded. I used the words. I could follow a demo. And I
could not have explained to a ten-year-old what was actually happening inside.

In April I watched Andrej Karpathy's talk at Sequoia AI Ascent. I nodded again. As I now realise, I understood
nothing.

I didn't say so to anyone. You can't, really. Not in a leadership role, not in public, not when everyone around
you seems fluent. So it was a relief to notice the very first question Karpathy was asked on that stage: a few
months earlier he had said he had never felt more behind as a programmer. His answer was that it felt
"a mixture of both", exhilarating and unsettling. If the man who named these things feels behind, the rest of us
are allowed to.

## Why reading more didn't help

I tried the usual route. More articles, more talks, more explainers. It made things worse, for three reasons.

The terms change faster than anyone can learn them. The explanations stay abstract: "the model reasons", "the
agent acts". The demos hide the thing I most wanted to know: which decisions were the AI's, and which were built in
by the people who made the demo. And the naive question, "but what did it actually do?", is exactly the one you
can't ask in a room full of people nodding.

## A private way out

So I built something nobody had to see. A game. Arkanoid, from 1988. A ball, a paddle, bricks, and a screen that
shows you everything. I am not a professional programmer, and that turned out not to matter: agents helped me build
the whole lab, the access to the game, the training pipeline, the checker. I described what I wanted in plain
language.

Then I tried to make an AI play.

**First, the way I thought AI worked.** Train a neural network using rewards. Claude, a large language model (LLM),
wrote all the training code for me. The network trained for about a day, 17 million steps. Its average score
stalled at 7–8 thousand points, and it never cleared all the bricks in round one. I tweaked the rewards. Nothing
moved.

**Then, the thing everyone kept calling an agent.** I gave a separate session of the same model a goal instead of a
method: here is the game, you can see the screen, press keys, move the mouse, jump back to saved moments; figure out
how to play and write a player. Nothing about how to play.

In under half an hour, in its own tests, its second version scored 101,820 points. After about 2.5 hours it had
written a player that averaged 174 thousand points in our separate referee's 20-minute tests, reaching rounds 14–19.

![Left: the neural network by millions of training steps. Right: the AI agent by hours of work](img/progress_en.png)

## The moment the words meant something

To aim, the agent needed to know how the ball bounces off the paddle. I didn't give it that rule. It loaded the
same falling-ball moment from a save state, placed the paddle at different offsets, and measured where the ball
went. It found six zones within ±12 pixels, then checked them again from another moment.

I watched this happen. Nobody had told it to run an experiment. It had decided what it didn't know, found a way to
measure it, and used the answer. That was the moment "agent" stopped being a word. A year of nodding, and it took a
ball and a paddle.

![Six paddle zones and where the ball goes from each, measured by the agent](img/bounce_en.png)

The second thing I saw was almost as useful. The agent's own tests looked fine. Then our referee, a separate program
that checks each game from the screen, found a game where the ball looped in round 25 forever. Activity, no
progress. The agent hadn't noticed. And the referee itself had to be fixed twice before we could trust it. I had
read a hundred times that AI needs oversight. Now I had seen what the sentence means.

Later, in round 33, the final boss turned out to shoot at the paddle. The agent had never seen anything shoot, and
the player lost all its lives in about 36 seconds. I noticed it on long runs and set a new task, "beat it", without
saying how. It did. In the end the player won 10 test games out of 10. And the finished player contains no AI at
all: it is ordinary code the agent wrote.

## What I actually got

Not the latest tool. Those will change again next month. What I got is a picture in my head that survives the
next new term: I have watched an agent try, measure, fail, get caught by a check, and fix itself when asked. Now
when someone says "agentic workflow" or "autonomous research", I know what it looks like when it works and what it
looks like when it quietly doesn't.

Counted at API prices, the research agent's model usage came to about $31, and the whole project, including the
failed training attempt, about $136. That is model usage, not my time. No team, no budget request, no one to
admit anything to.

The honest limits: this was not a fair contest between methods, the agent also had better controls and save
states, and we never finished improving the neural network. The game is deterministic. We built the game tools and
the referee ourselves, and a human started every fix. I can't tell what the model already knew about Arkanoid.

## If you've nodded too

You don't have to confess anything. Pick a small problem you can watch: a game, a toy version of your own queue,
anything with a visible result. Ask an agent to figure it out. Then watch what it tries, what makes it change
course, and what your check catches that it missed. An afternoon of that taught me more than a year of nodding.

What was the last AI term you nodded at without understanding? I'd genuinely like to know I'm not the only one.

Code: [neuronoid](https://github.com/askarzh/neuronoid) and
[neuronoid-strategist](https://github.com/askarzh/neuronoid-strategist); the agent's log is
[LOG.md](https://github.com/askarzh/neuronoid-strategist/blob/main/LOG.md) (in Russian).
