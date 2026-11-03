Most of us judge AI agents from demos. I'm not a professional programmer, so I built a small lab instead: a 1988
game I could watch.

First, Claude wrote me the code to train a neural network to play Arkanoid. After about a day of training it never
cleared all the bricks in round one.

Then I gave Claude a goal instead of a method: here is the screen, the mouse and save states; figure out how to play
and write a player. It chose its own experiments. It measured how the ball bounces off the paddle, a rule I didn't
give it. After a few more tasks, its player won all ten test games, checked by a separate referee program.

Two things I could finally see, and now look for in business:

1. A metric can rise while the job isn't done. My network's score kept improving, partly from bonus capsules, and it
still never cleared the wall.
2. The builder's tests aren't enough, and neither is the checker. The referee caught a loop the agent missed, and
the referee itself had to be fixed twice.

The part I like most: agents also helped me build the lab, the training pipeline, the game access and the referee.

If agents feel abstract, find your own Arkanoid and watch what the agent tries.

The full story is in the first comment.
