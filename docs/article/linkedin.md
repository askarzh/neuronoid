# I finally understood Andrej Karpathy, thanks to Arkanoid!

![The final screen of DOS Arkanoid: the VAUS ship escapes after the DOH boss is destroyed](img/final.png)

This is the last screen of Arkanoid for DOS, released in 1988. The ship escapes because the final boss, DOH, has
been destroyed in round 33. No API, no access to the game's memory: only a 320×200 picture, a keyboard and a mouse.

A program written by AI reached this screen in 10 games out of 10. The AI itself did not play. It studied the game
by looking at the screen and wrote an ordinary player program, and that program played. Claude and I built the
access to the game and a separate referee, and I set the tasks.

Six months ago, in April 2026, I watched [Andrej Karpathy's talk](https://youtu.be/96jN2OCOfLs) at Sequoia AI
Ascent. In Software 1.0 you write code. In Software 2.0 you are "programming by creating datasets and training
neural networks". In Software 3.0 "your programming now turns to prompting", and the context window is "your lever
over the interpreter that is the LLM". I watched, nodded and, as I now realise, understood nothing.

I am not a professional programmer. Vibe coding, agents, new paradigm: for me these had become abstractions you
can't touch. Recently I watched the talk again and was surprised that I understood almost every word. In those six
months I had lived it.

## Before: Software 2.0

To me, "build an AI that plays a game" meant reinforcement learning: a neural network looks at the screen, presses
buttons, gets a reward for points and learns over millions of attempts.

The agent trained for about a day on a laptop CPU: 17 million steps. Its average score got stuck at 7–8 thousand
points, and it never cleared the first round of bricks. I kept tweaking the reward. Each change helped a little,
but it went no further.

All that code was written by Claude, a large language model (LLM), as my programming assistant, inside the method
I had chosen. To me, AI was the trained network and the LLM was a tool for building it. That is Software 2.0.

## After: Software 3.0

Then a separate session of the same model got a body and a goal instead of a coding task. The body was an emulator
that boots the game from scratch: the researcher could look at the screen, press keys, move the mouse and jump back
to saved moments. The goal: "figure out how to play and write a player program". Nothing about how to play.

The model was the same, Claude Opus 5.5. Its role changed, and so did the conditions: precise mouse control, the
full frame, save states for experiments.

Here is what I understood on the second viewing. My program was no longer code but a one-page text. The LLM read
it, the very interpreter Karpathy talked about, and wrote a plain code player that needs no LLM to play. Software
3.0 wrote Software 1.0.

![Left: the RL agent by millions of training steps. Right: the LLM researcher by hours of work](img/progress_en.png)

In under half an hour its second version scored 101,820 points and reached round 9 without losing a life. In about
2.5 hours it averaged 174 thousand points and rounds 14–19 in 20 minutes of play under our referee.

## The moment that surprised me

To aim, the researcher tested what decides the bounce. It loaded the same falling-ball moment from a save state,
placed the paddle at different offsets and measured how the ball flew off. It found six zones within ±12 pixels,
from a shallow bounce left to a shallow bounce right.

I did not expect an LLM to realise this was physics and to decide on its own to measure paddle offset, ball speed
and bounce angle. Whether it knew Arkanoid bounces from training, we can't tell. But it designed and ran the
experiment itself. This was no longer an assistant writing code on my hint. It was a researcher.

![Six paddle zones and where the ball goes from each, measured by the researcher](img/bounce_en.png)

Then came fixes. In round 33 the boss shoots, and the player lost all its lives in about 36 seconds. I set a new
task, "beat DOH", without saying how, and the researcher taught the player to dodge. Later the ball got stuck in
round 25; the researcher replayed it 24 times, found 4 trapped cases and added an escape move.

## How it was verified

Karpathy's other point clicked here: LLMs "can easily automate what you can verify". A separate referee boots the
game, waits a random number of frames, hands control to the player and reads the result from the screen. Final
run: 10 wins over DOH out of 10, mean score 343,977. At API prices the researcher's work would have cost about
$31, the whole project about $136.

## From Arkanoid to business

AI agents in business used to be an abstraction for me. Now I have a few rules I saw with my own eyes.

**Give the agent a sandbox where it can fail.** The researcher rewound again and again. At work that means a copy
first, say a CRM copy with no access to real payments.

**Decide how to check before you automate.** "10 out of 10" means something only because a separate referee
counted. In business it can be a test that reconciles the final balance. If you can't say how you'll know the
agent succeeded, it's too early.

**Make the LLM part of the system, not a helper on the side.** Don't just give staff a chatbot. Build the LLM into
the process where someone has to make sense of something new, and keep ordinary code and checks around it.

**If it didn't work, that's not the final word.** Six months ago I put this idea aside. Now the same kind of task
took a few hours. When something with AI fails, I come back to it a few months later.

**Questions I would now ask a vendor:** How do you check the result independently of whoever built it? At which
steps does the model run, and what is done by a program you can read and test? When a new failure appears, who
notices it and who sets the task to fix it?

## Limitations

The game is deterministic. We built the body and the referee. A human started every fix. The main repo was
off-limits by the task's rules, not technically, and the log shows the researcher never went there.

## Instead of a conclusion

Karpathy's words went past me six months ago. I understood them not from a lecture but from Arkanoid, watching the
ball fly exactly where a player written by an LLM was aiming.

If new-paradigm talk feels too abstract, find your own Arkanoid: a simple task where you can see the result with
your own eyes.

Code: [neuronoid](https://github.com/askarzh/neuronoid) and
[neuronoid-strategist](https://github.com/askarzh/neuronoid-strategist); the researcher's log is
[LOG.md](https://github.com/askarzh/neuronoid-strategist/blob/main/LOG.md) (in Russian).
