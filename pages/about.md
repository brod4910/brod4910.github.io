---
layout: page
title: About
permalink: /about/
weight: 3
---

# **About Me**

Hi I am **{{ site.author.name }}** :wave:,<br>
A Machine Learning Engineer driven by none other than my two cats and dog.

{% include elements/figure.html image="https://i.imgur.com/qPyWq7C.jpg" caption="Pepper" %}

I'm a generally outgoing and silly person but when it comes down to it, I'm ready to tackle almost any challenage except training another dog.

{% include elements/figure.html image="https://i.imgur.com/glcIOf8.gif" caption="Luna... a mind of her own and the will power to break a human. I love her." %}

<div class="row">
{% include about/skills.html title="Programming Skills" source=site.data.programming-skills %}
{% include about/skills.html title="Other Skills" source=site.data.other-skills %}
</div>

<div class="row">
{% include about/timeline.html %}
</div>