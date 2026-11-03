# I'm not a programmer. A 1988 game taught me what AI agents really are

![The final screen of DOS Arkanoid: the VAUS ship escapes after the DOH boss is destroyed](img/final.png)

This is the last screen of Arkanoid, a DOS game from 1988. The ship escapes because the final boss has been
destroyed in round 33. A program written by an AI agent reached this screen in 10 test games out of 10. The agent
studied the game from a 320×200 picture on the screen and wrote an ordinary player program. The finished player
needs no AI to play.

I am not a professional programmer. I followed the agent demos, but I couldn't tell which decisions the model was
making and which were built into the setup. So I built a small lab where I could see it for myself. The surprising
part: agents helped me build the lab too. I wouldn't have known how to build it on my own.

## Why a game

In a business pilot it is hard to separate what the agent did from everything else changing around it. In a game
I could repeat the same situation, change one action and see the result. The ball either flies where the player
aimed, or it doesn't. And I could see when a higher score did not mean the job was done.

## Two ways to build an AI player

**The classic way.** I first did what "AI that plays a game" used to mean: train a neural network using rewards.
Claude, a large language model (LLM), wrote all the training code for me. The network trained for about a day,
17 million steps. Its average score stalled at 7–8 thousand points, and it never cleared all the bricks in round
one. Once, its best score turned out to come from catching bonus capsules rather than breaking the wall.

**The agent way.** Then I gave a separate session of the same model a different kind of job. Not a method to
implement, but a goal: here is the game, you can see the screen, press keys, move the mouse and jump back to saved
moments; figure out how to play and write a player. Nothing about how to play.

In the first setup I chose the method and Claude implemented it. In the second, Claude chose its own experiments,
read the results and decided what to try next. That loop is what made it an agent for me.

In under half an hour, in its own tests, its second version scored 101,820 points. After about 2.5 hours, the player
averaged 174 thousand points in our referee's 20-minute tests, reaching rounds 14–19.

![Left: the neural network by millions of training steps. Right: the AI agent by hours of work](img/progress_en.png)

## The moment it clicked

To aim, the agent needed to know how the ball bounces off the paddle. I didn't give it that rule. It loaded the same
falling-ball moment from a save state, placed the paddle at different offsets and measured where the ball went. It
found six zones within ±12 pixels, then checked them from another moment.

It was not answering a question. It was finding out what it didn't know, by experiment, and then using the answer.
Andrej Karpathy says that with LLMs "your programming now turns to prompting". Watching the agent design its own
measurement is what made that sentence real for me.

![Six paddle zones and where the ball goes from each, measured by the agent](img/bounce_en.png)

## The business problems I could finally see

I don't think a company is a game. But the game let me see, in one small world, problems that business pilots
usually hide.

**A metric can rise while the job isn't done.** My neural network kept improving its score and still never cleared
the first wall. Points came easier from bonus capsules than from the actual task. So for the final test, success was not a number of points: the referee recognised victory by the ending screen itself. That is the question I would now ask about any agent's KPI:
what exactly counts as finished?

**The builder's tests are not enough, and neither is the checker.** The agent's own tests looked fine. A separate
program, our referee, then found a game where the ball looped in round 25 forever: lots of activity, no progress.
The player even had a rule for detecting stagnation; it was fooled by signals that looked like progress. And the
referee itself had to be fixed twice before we could trust it. In business terms: an acceptance check separate from
whoever built the solution, and evidence that the check itself works.

**Some failures sit outside everything you tested.** In round 33 the boss fires at the paddle. The agent had never
seen anything shoot, and the player lost all its lives in about 36 seconds. I noticed it on long runs and set a new
task, "beat it", without saying how. The agent found the projectiles in the frames and taught the player to dodge.
Someone still has to notice the new kind of failure and ask.

**Screen-only systems are not out of reach.** The agent worked only from what was visible and the actions it was
allowed. Many companies run legacy software with nothing but a screen. A copy isolated from live systems would play
the role of my save states, though only as an approximation: real processes have side effects a game doesn't.

## The part I find most interesting

Agents let me build a lab I couldn't have built myself: the training pipeline, the access to the game, the referee
and the researcher. Counted at API prices, the research agent's model usage came to about $31, and the whole project,
including the failed training attempt, about $136. That is model usage, not my time.

What I got for it was not a playbook. It was the ability to see what an agent tries, which evidence changes its
mind, and which failures my checks missed.

## Limitations

This was not a fair contest between methods: the agent also got precise mouse control, the full frame and save
states, and we never finished training an improved neural network. The game is deterministic. We built the game
tools and the referee, and a human started every fix. We can't tell what the model already knew about Arkanoid.

## Find your own Arkanoid

Pick a small problem where you can see the result. Then watch three things: which experiment the agent chooses,
what evidence makes it change course, and which failure your check doesn't catch.

What could your team learn from watching an agent work, rather than from another demo?

Code: [neuronoid](https://github.com/askarzh/neuronoid) and
[neuronoid-strategist](https://github.com/askarzh/neuronoid-strategist); the agent's log is
[LOG.md](https://github.com/askarzh/neuronoid-strategist/blob/main/LOG.md) (in Russian).
