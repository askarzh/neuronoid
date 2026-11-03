I finally understood Andrej Karpathy, thanks to Arkanoid!

Six months ago I watched his talk about Software 1.0, 2.0 and 3.0. I nodded and understood nothing. I'm not a
professional programmer, and words like agents and new paradigm felt like abstractions you can't touch.

Then I ran a small experiment. First I tried to teach an AI to play Arkanoid, the 1988 DOS game, by training a
neural network. About a day of training, 17 million steps, and it never cleared the first round.

Then I gave the same model, Claude, a different job. Not "write training code" but: here is the game, the screen,
the mouse and save states. Figure out how to play and write a player.

What surprised me most: it decided on its own to measure how the ball bounces off the paddle, like a physics lab.
Then it wrote an ordinary program that beat the final boss in 10 out of 10 games, checked by a separate referee.

That's when Karpathy's words clicked. My program was a one-page text. The LLM read it and wrote plain code. Software
3.0 wrote Software 1.0.

What I took for business: give agents a sandbox, decide how to check before you automate, and build the LLM into
the process, not next to it.

Find your own Arkanoid. It explains more than any lecture.

Full story: <link>
