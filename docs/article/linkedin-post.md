I finally understood Andrej Karpathy, thanks to Arkanoid!

A day of neural network training couldn't clear the first screen of a 1988 game. Then an LLM figured the game out
by itself and wrote a player that beat it.

Six months ago I watched Karpathy's talk about Software 1.0, 2.0 and 3.0. I nodded and understood nothing. I'm not a professional programmer, and words like "agents" were just words to me.

Claude wrote the training code for me, and a neural network learned to play for
about a day: 17 million steps. It never cleared all the bricks in round one.

This time I asked Claude to investigate the game itself: here is the screen, the mouse and save states, figure out
how to play and write a player.

What surprised me most: it designed its own experiment to measure how the ball bounces off the paddle. After more
tasks and fixes, its ordinary program beat the final boss in 10 out of 10 games, checked by a separate referee.

Then I watched the talk again, and it finally clicked. I had started with a one-page assignment. The LLM read it
and wrote plain code. Software 3.0 wrote Software 1.0.

My business takeaway: give agents a sandbox, decide how to check before you automate, and give the LLM a defined
role in the process, with tools and checks.

The full story, with the experiments and the questions I'd now ask a vendor, is in the first comment.
