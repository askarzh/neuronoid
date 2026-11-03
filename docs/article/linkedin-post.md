An AI agent is only as good as the system around it. I learned that from a 1988 game.

A neural network I trained for about a day couldn't clear the first round of Arkanoid. Then I gave the same model,
Claude, a different role: not my coding assistant, but a researcher. Its job: figure the game out from the screen and
write a player program.

That program beat the final boss in 10 games out of 10. Three things made the difference:

1. A sandbox. With save states it could run experiments, and it measured six bounce zones on the paddle by itself.
2. An independent referee. A separate program checked every game, and it caught a bug the researcher's own tests had
   missed.
3. Code as the output. The finished player has no LLM inside: it's fast, readable and testable. Software 3.0 wrote
   Software 1.0.

And one human: me, noticing new kinds of failure and setting the next task.

My takeaway for business: before you automate with an agent, decide where it can fail safely, who checks the result,
and what ordinary code does the actual work.

The full story, with the experiments and the questions I'd now ask a vendor, is in the first comment.
