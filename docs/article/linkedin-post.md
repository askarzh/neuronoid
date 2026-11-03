I'm not a programmer. Yet I just ran an AI agent experiment that taught me more about agents than any talk.

The lab: Arkanoid, a 1988 DOS game. A neural network I trained for about a day never cleared round one. Then I gave
an AI agent, Claude, a goal instead of instructions: here is the screen, the mouse and save states; figure out how
to play and write a player.

It ran its own experiments. It measured how the ball bounces off the paddle, a rule I never told it. After a few more
tasks, its player beat the game in 10 out of 10 games, checked by an independent referee.

What struck me: the game is a business in miniature.

- Only a screen, no internals: your legacy systems and customers.
- The score: your KPI.
- The hidden bounce rule: the tacit rules nobody wrote down.
- Save states: a safe copy to experiment on.
- The boss that suddenly shoots: the event your process has never seen.
- The referee: your audit.

And the twist: I could only build this lab because of agents. I described it in plain language.

If agents feel abstract, find your own Arkanoid: a small problem where you can see the result.

The full story is in the first comment.
