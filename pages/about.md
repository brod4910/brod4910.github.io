---
layout: page
title: About
permalink: /about/
weight: 3
---

# **About Me**

Hi I am **{{ site.author.name }}** :wave:,<br>
A Machine Learning Engineer driven by none other than my two cats and dog.
I'm a generally outgoing and silly person but when it comes down to it, I'm ready to tackle any challenage except training another dog.

<div class="row">
{% include about/skills.html title="Programming Languages" source=site.data.programming-languages %}
{% include about/skills.html title="Engineering Skills" source=site.data.eng-skills %}
{% include about/skills.html title="Spoken Languages" source=site.data.languages %}
{% include about/skills.html title="Other Skills" source=site.data.other-skills %}
</div>

<div class="row">
{% include about/timeline.html %}
</div>

<div style="display: flex; justify-content: space-between;">

  <div style="flex: 1; margin-right: 10px;">
    {% include elements/figure.html image="https://i.imgur.com/qPyWq7Cm.jpg" caption="Pepper, my curious, snaggle-toothed boy." %}
  </div>

  <div style="flex: 1; margin-left: 10px;">
    {% include elements/figure.html image="https://i.imgur.com/RAfag30m.jpg" caption="DK (Dumpster Kitty), my girl with attitude." %}
  </div>

</div>


{% include elements/figure.html image="https://i.imgur.com/glcIOf8.gif" caption="Luna... a mind of her own and the will power to break any human. I love her. Enjoy this footage of her being really obidient... She really is a great dog, and training her has been the hardest thing I've done to date." %}
