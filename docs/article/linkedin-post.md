Six months ago I tried to build an AI that plays Arkanoid — the 1988 DOS game, from screen pixels only. I failed.

This time it beat the game: all 33 rounds and the final boss, in 10 out of 10 games checked by an independent
referee.

What changed? Not just the models.

First I went the classic way: reinforcement learning. About a day of training on a laptop CPU, 17 million steps —
and the agent never honestly cleared round one. An LLM wrote all that RL code for me. To me it was an assistant.

Then I gave the same model, Claude Opus 5.5, a different role. Not "write training code", but: here is a body —
screen, keys, mouse, save states. Figure out how to play.

It ran experiments like a scientist: measured how the ball bounces off the paddle, found six zones, wrote a player
in plain readable code, and fixed it when I set new tasks. Cost at API prices: about $31.

The lesson for anyone running technology: an AI system is not only a trained model. The LLM can be the part that
understands the problem and builds the solution. And conclusions like "AI can't do this" go stale in six months.

Honest limits, the research loop and questions to ask vendors — in the article: <link>
