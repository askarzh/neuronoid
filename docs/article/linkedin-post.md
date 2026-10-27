Six months ago I tried to build an AI that plays Arkanoid — the 1988 DOS game, from screen pixels only. I failed.

This time it reached round 33 and beat the final boss in 10 out of 10 games, checked by a separate referee
program we wrote.

What changed? Not just the models.

First I went the classic way: reinforcement learning. About a day of training on a laptop CPU, 17 million steps —
and the agent never honestly cleared round one. (Not a fair contest: RL trained only briefly.) An LLM wrote all
that RL code for me. To me it was an assistant.

Then I gave the same model, Claude Opus 5.5, a different role. Not "write training code", but: here is a body —
screen, keys, mouse, save states. Figure out how to play.

It ran experiments like a scientist: measured how the ball bounces off the paddle, found six zones, wrote a player
in plain readable code, and fixed it each time I set a new task. The researcher's work would have cost about $31
at API prices — an estimate, since I used a subscription.

The lesson for anyone running technology: an AI system is not only a trained model. The LLM can be the part that
understands the problem and builds the solution. And conclusions like "AI can't do this" are worth re-checking
every six months.

Honest limits, the research loop and questions to ask vendors — in the article: <link>
