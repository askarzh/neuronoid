# I finally understood Andrej Karpathy, thanks to Arkanoid!

![The final screen of DOS Arkanoid: the VAUS ship escapes after the DOH boss is destroyed](img/final.png)

This is the last screen of Arkanoid for DOS, released in 1988. The ship escapes because the final boss, DOH, has
been destroyed in round 33. No API, no access to the game's memory: only a 320×200 picture, a keyboard and a mouse.

A program written by AI reached this screen in 10 games out of 10. The finished player ran without any LLM. The AI studied the game from the screen and wrote an ordinary program that played. Claude and I
built the tools for interacting with the game and a separate program to check the results, and I set the tasks.

Six months ago, in April 2026, I watched [Andrej Karpathy's talk](https://youtu.be/96jN2OCOfLs) at Sequoia AI
Ascent. In Software 1.0 you write code. In Software 2.0 you are "programming by creating datasets and training
neural networks". In Software 3.0 "your programming now turns to prompting", and the context window is "your lever
over the interpreter that is the LLM". I watched, nodded and, as I now realise, understood nothing.

I am not a professional programmer. Vibe coding, agents, a new paradigm: I knew the words, but I didn't really
know what they meant in practice. Recently I watched the talk again and was surprised that I understood almost
every word. Now I had my own experience to connect them to.

## Before: Software 2.0

I went the classic way, reinforcement learning: a neural network looks at the screen, gets a reward for points and learns over millions of attempts.

The agent trained for about a day on a laptop CPU: 17 million steps. Its average score stalled at 7–8 thousand
points, and it never cleared all the bricks in round one. I kept tweaking the reward. The tweaks helped a little,
but I couldn't get past that plateau.

That attempt was Software 2.0: the program was the network's learned weights. Claude, a large language model
(LLM), wrote all the training code as my programming assistant, inside the method I had chosen. To me, AI was the
trained network and the LLM was a tool for building it.

## After: Software 3.0

Then I started a new session with the same model and gave it a goal and tools to explore the game instead of a
coding task. Through an emulator, it could look at the screen, press keys, move
the mouse and jump back to saved moments. The goal: "figure out how to play and write a player program". Nothing
about how to play.

The model was the same, Claude Opus 5.5. Its role changed, and so did the conditions: precise mouse control, the
full frame, save states for experiments. So this is not a fair contest between methods: we never finished
training a reworked RL setup.

Instead of specifying a training method, I gave it a one-page
assignment. The LLM was the interpreter Karpathy had described. It read my instructions and wrote a player in
ordinary code, which could run without an LLM. Software 3.0 wrote Software 1.0.

![Left: the RL agent by millions of training steps. Right: the LLM researcher by hours of work](img/progress_en.png)

In under half an hour, in its own tests, its second version scored 101,820 points and reached round 9 without
losing a life. After about 2.5 hours it had built a player that averaged 174 thousand points in our referee's
20-minute tests, reaching rounds 14–19.

## The moment that surprised me

To work out where to aim, it tested what determines the ball's bounce. It loaded the same falling-ball moment from
a save state, placed the paddle at different offsets and measured how the ball flew off. It found six zones within
±12 pixels, from a shallow bounce left to a shallow bounce right.

I did not expect an LLM to realise this was physics and to design its own experiment measuring paddle offset, ball
speed and bounce angle. We can't tell what it already knew about Arkanoid from its training. But it designed and
ran the experiment itself. It wasn't just writing code based on my suggestions anymore. It was a researcher.

![Six paddle zones and where the ball goes from each, measured by the researcher](img/bounce_en.png)

Then came further tasks and fixes. In round 33 the boss shoots back, and at first the player lost all its lives in about 36
seconds. I set a new task, "beat DOH", without saying how, and the researcher taught the player to dodge. Later
the ball kept looping in round 25 without making progress. The first fix didn't hold. The second time the
researcher replayed that round 24 times from two saved moments, found 4 trapped cases and added an escape move.

## How it was verified

Karpathy's other point clicked here: LLMs "can easily automate what you can verify". A separate referee boots the
game, waits a random number of frames, hands control to the player and reads the result from the screen. Final
run: ten games with different start delays, 10 wins over DOH, mean score 343,977.

At API rates, model usage would have been about $31 for the researcher and about $136 for the whole
project. It's an estimate of model usage, not a bill.

## From Arkanoid to business

Here are the practical rules I took from this project.

**Give the agent a sandbox where it can fail.** At work I would start with a copy, say a CRM copy with no access to real payments.

**Decide how to check before you automate.** "10 out of 10" was checked by a separate program, not reported by the
researcher itself. In business that check could be a test that reconciles the final balance. If you can't say how
you'll know the agent succeeded, it's too early.

**Give the LLM a defined role in the system.** In my first attempt the LLM was a helper on the side; in the second it was part of the system. What I would try now: instead of just giving staff a chatbot, build the LLM into the process where something new has to be figured out, with code and checks around it.

**If it didn't work, that's not the final word.** Six months ago I put this idea aside. Now, when something with AI fails, I come back to it a few months later.

**Questions I would now ask a vendor:**

- How do you check the result independently of whoever built it?
- Where in the process is the model used, and what does ordinary, testable code handle?
- When a new kind of failure appears, who notices it and who sets the task to fix it?

## Limitations

The game is deterministic. We built the game tools
and the referee. A human started every fix. Web search and files outside its folder were off-limits by the task's
rules, not technically, and the log shows the researcher never went there.

## What finally clicked

Six months ago I heard Karpathy's words without really understanding them. I understood them from Arkanoid,
watching the ball fly exactly where a player written by an LLM was aiming.

If new-paradigm talk feels too abstract, find your own Arkanoid: a simple task where you can see the result with
your own eyes.

Code: [neuronoid](https://github.com/askarzh/neuronoid) and
[neuronoid-strategist](https://github.com/askarzh/neuronoid-strategist); the researcher's log is
[LOG.md](https://github.com/askarzh/neuronoid-strategist/blob/main/LOG.md) (in Russian).

Where in your work could an LLM write the rules, instead of you training a model to learn them?
