# I'm not a programmer. A 1988 game taught me what AI agents really are

![The final screen of DOS Arkanoid: the VAUS ship escapes after the DOH boss is destroyed](img/final.png)

This is the last screen of Arkanoid, a DOS game from 1988. The ship escapes because the final boss has been
destroyed in round 33. A program written by an AI agent reached this screen in 10 games out of 10, looking only at
the screen, a 320×200 picture.

I am not a programmer. I ran this experiment by giving instructions in plain language. And that is the point of this
article: a year ago someone like me could not have built this lab at all. Doing it is what finally made AI agents
concrete for me.

## Why a game

Talk about AI agents in business is everywhere, and most of it stays abstract. Part of the reason is that business
results are invisible and slow. You deploy something and wait weeks to learn whether it helped.

A game is the opposite. There is a score. Feedback is instant. Failure is safe. You can rewind. And you can literally
watch the AI think: the ball either flies exactly where the player aimed, or it doesn't. For learning what agents can
do, a game is what a flight simulator is to a pilot.

## Two ways to build an AI player

**The classic way.** I first did what "AI that plays a game" used to mean: train a neural network on rewards. Claude, a
large language model (LLM), wrote all the training code for me. The network trained for about a day, 17 million
steps. Its average score stalled at 7–8 thousand points, and it never cleared all the bricks in round one.

**The agent way.** Then I gave a separate session of the same model a different kind of job. Not "write this code",
but a goal: here is the game, you can see the screen, press keys, move the mouse and jump back to saved moments;
figure out how to play and write a player. Nothing about how to play. That is the difference between an assistant
and an agent: an assistant does the step you describe, an agent works out the steps itself.

In under half an hour, in its own tests, its second version scored 101,820 points. After about 2.5 hours, the player
averaged 174 thousand points in our independent referee's 20-minute tests, reaching rounds 14–19.

![Left: the neural network by millions of training steps. Right: the AI agent by hours of work](img/progress_en.png)

## The moment it clicked

To aim, the agent needed to know how the ball bounces off the paddle. I didn't know that rule, and I didn't tell it.
It loaded the same falling-ball moment from a save state, placed the paddle at different offsets and measured where
the ball went. It found six zones within ±12 pixels, then checked them from another moment.

That was the moment. It was not answering a question. It was finding out what it didn't know, by experiment, and
then using the answer. Andrej Karpathy describes this new way of programming as "your programming now turns to
prompting". Watching the agent run its own experiment is what made that sentence real for me.

![Six paddle zones and where the ball goes from each, measured by the agent](img/bounce_en.png)

## Arkanoid is a business in miniature

The longer I watched, the more the game looked like a company:

- **Only a screen, no access to the internals.** Like a legacy system, a customer or a competitor: you see outputs,
  not the logic behind them.
- **The score.** Your KPI.
- **The hidden bounce rule nobody wrote down.** The tacit rules in every company: why some invoices get rejected,
  why some customers leave.
- **Save states.** Replaying historical data, or a staging copy where mistakes are free.
- **The boss that suddenly shoots.** In round 33 the boss fires at the paddle, something the agent had never seen,
  and the player lost all its lives in about 36 seconds. A new kind of event your process has never met.
- **The ball stuck looping in round 25.** A process that runs forever without progress, and nobody inside notices.
- **The referee.** A separate program that checked every game, like an audit. It caught the loop that the agent's own
  tests had missed.

What the agent did in the game is what I would now try with an agent in a business: observe a system from the
outside, find the undocumented rules by experiment, and let an independent check decide whether it worked. In the
game a human was still needed: I noticed the shooting boss and set a new task, "beat it", without saying how. That
part doesn't go away in business either.

This is an analogy, not proof. A game is deterministic, rewindable and cheap to replay; most business processes are
not. But that is exactly why I would start with a visible proxy before betting on an invisible process.

## The part I find most interesting

I could only run this experiment because of agents. A non-programmer built a research setup with a training
pipeline, an emulator harness, an independent referee and a researcher agent, by describing what he wanted. At API
prices, the agent's research would have cost about $31, and the whole project, including the failed training
attempt, about $136.

So the cheapest way for a business leader to understand AI agents may not be another talk or report. It may be to
run a small experiment yourself, on something you can see.

## Limitations

This was not a fair contest between methods: the agent also got precise mouse control, the full frame and save
states, and we never finished training an improved neural network. We built the game tools and the referee, and a
human started every fix. We can't tell what the model already knew about Arkanoid from its training.

## Find your own Arkanoid

Pick a small problem where the result is visible: a game, a simulation of your queue, a replay of last month's
orders. Give an agent a goal, a safe place to experiment and an honest referee, and watch what it does.

What visible experiment would teach your team more about agents than the next presentation?

Code: [neuronoid](https://github.com/askarzh/neuronoid) and
[neuronoid-strategist](https://github.com/askarzh/neuronoid-strategist); the agent's log is
[LOG.md](https://github.com/askarzh/neuronoid-strategist/blob/main/LOG.md) (in Russian).
