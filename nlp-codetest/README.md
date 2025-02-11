# Seedtag Codetest:  NLP Researcher
<p style="text-align:center;">
<img src="Seedtag.png" alt="Seedtag" height="10%" width="10%">
</p>

===== start of transmission =====

**Agent Kallus**,

If you are reading this message, you have reached the safe house that The Empire has installed near the rebel base in the city of Norg Bral. Well done, although your work here has only just begun. As you know, the planet Mandalore is of great strategic importance to us and we have to be prepared for any movement by the Rebel commandos stationed there. To anticipate them, we have been developing in recent months a communications interception system that we are close to launching and that is installed at their location. The next module that needs to be developed is the transmission classification subsystem. This is where you come in.

## Mission details

Our intelligence department has classified all intercepted communications over the last few weeks into 7 different categories, based on the Rebel divisions they belong to. Your mission consists of three parts:

### *Part 1*

The first part of your mission is to improve the provided baseline classification system ([```classifier_baseline.ipynb```](./part1/classifier_baseline.ipynb)) so that the operations department can evaluate threats faster and more confidently. However, according to intelligence some texts from one of the categories contain noise. Thus, you should take care of noisy data when improving the baseline model.


### *Part 2*

Recently, we have detected what it seems to be a new category among the intercepted messages, and we have gathered a few of them. The second part of your mission consists of creating a system smart enough to learn, from previous categories data or general knowledge, how to classify messages even when it has been trained with just a few examples.

### *Part 3 (OPTIONAL MISSION)*

For the third and final part of your mission, you need to build a message matcher system, that is able to detect from a set of samples, the ones most similar to a given message according to their content. (for a reference, see [```matcher_baseline.ipynb```](./part3/matcher_baseline.ipynb)).


## Delivery instructions

All solutions and corresponding analysis and evaluation, as well as the hacked baseline model, should be provided in a jupyter notebook named ```<username-en-github>_codetest4_seedtag``` and send it attached in an email to alliance@seedtag.com. In case your mail system does not allow you to send the file, you can rename the extension to .a.

Good luck on your mission and may the force be with you.


===== end of transmission =====


