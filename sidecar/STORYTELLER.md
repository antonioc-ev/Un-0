# As far as the sidecar storyteller got

The target was a storyteller. The oscillator bag would be the prose engine.
A story would keep a character, a dilemma, and a resolution, and the ending
would still be about those. The arc is that relation: who, what the trouble
was, and whether the last sentence names them. The arc is a reading. It was
never a loss.

The bag we can train is an oscillator with next-word loss. The word enters as
a phase offset inside the coupling. The dynamics learn slowly. The readout is
narrow. That is the plain tug. Its best run is loss 1.715 at epoch 221, in
`sidecar/data/checkpoints_n256_r64_slow32_fullstory/best.safetensors`. Free
generation from that bag is locally word-like and does not hold a plot. After
“birdcage” it went to “scary sky,” cake, and scale.

## What we tried, so the story would stay a story

Character, dilemma, and resolution were drives on the slow wheels, not a menu
of story words. The character stayed on for the whole story. The dilemma
turned on at its word and held lightly, then faded when the last sentence
began. The resolution pulled harder through that last sentence. Stamp wheels
sat still until `.!?`, then took one full step. One arc wheel stepped only
while the resolution drive was on. The latch stayed off. Sentence periods did
not reset the bag. The end piece did.

Stronger drives beat weaker ones. Relaxing them (pull 1, character 1, dilemma
0.1, resolution 1) raised the best loss from 1.715 to 1.894 and the samples
got worse. Hit-only dilemma and resolution were worse. Halving the character
drive collapsed to “bake bake bake.”

We also tried to make order a property of the wheels, from the sequential
Kuramoto memory: a route matrix, lagged recognition, and one coupling per
stored pose. A pose spring, a gain on the existing tug, random poses, and an
online “click into a slot” write all failed. The write filled the slots while
the loss was still around 6 to 8, then the loss rose. The plain tug, with the
route left out of the step, is the run that still learns.

A repetition tax and a one-time half-step of the word wheels were decode
attempts to keep the last lines cogent. The tax was flipping one-point leads
(friends over cake, the bird over scale). It did not create the missing
sentence. The half-step did not make the first free word the period. That
period is about ten points and more than a hundred ranks behind “scary.”

## What actually closes the Mia story

The shelf is a retrieval index, built with BM25 when the training stories are
loaded and saved at `sidecar/data/bm25_shelf.json`. If the words so far are a
long exact suffix of one kept story, the next kick is the piece that followed
that window. The wheels are turned by that kick. Temperature does not get a
vote. For shelf story 0 that continuation is the stored one, through “They
played and had fun all day.”

That is a retrieval engine. A bare “Once upon a time” does not lock, because
many stories share the opening. At a warmer temperature the wheels may pick
another piece from the retrieved sentence. They do not keep the order. The
sample becomes “birdcage.,. the birdcage, little bird.”

## Why this is not a storyteller

Two oscillator ideas get confused here.

Sequential Kuramoto stores patterns and plays them back. The order lives in
the route. The oscillators do not invent the next sentence. The shelf lock is
that idea in words. It matches the philosophy, and it only repeats a story
that was already stored.

An oscillator language model is a different claim. Those models are trained on
a long stream, such as Tiny Shakespeare, and scored in bits per character.
Local fluency counts. A line can wander and still look like the corpus. They
still trail a gated recurrent net. They do not finish one particular plot.

Our plain tug is that second machine, trained on 317 short stories that each
reset at the end piece. Loss 1.715 says the next piece is often nearby. It
does not say that forty pieces later the bag still knows the bird got out and
they played all day. We do not have a side objective that teaches the wheels
to speak a retrieved line in order, and the oscillator’s own scores do not do
it. Storytelling is not in this bag. Retrieval can replay a stored story.
Closing the gap from there to a teller that makes a new one is a different
tool than the one we have.
