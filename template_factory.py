import re
import random

class NLIConverter:
	def __init__(self, mask_token = "<mask>"):
		positive = ['positive', 'yes', 'entail', 'entails', 'entailment', 'true', 'affirmative', 'yep']
		neutral = ['neutral', 'maybe', 'maybe not', "don't know", 'not necessarily', 'not sure', 'uncertain']
		negative = ['negative', 'no', 'non-entailment', 'nonentailment', "doesn't entail", 'instead', 'false', 'na']
		self.mask_token = mask_token
		self.verbalizer_pool = [positive, neutral, negative]
		self.template_pool = [f"<premise> ? {self.mask_token} <hypothesis>",
							  f"does <premise> mean that <hypothesis> ? {self.mask_token}",
							  f"does <premise> indicates <hypothesis> ? {self.mask_token}",
							  f"does <premise> implies <hypothesis> ? {self.mask_token}",
							  f"<premise> and <hypothesis> are in a {self.mask_token} relationship",
							  f"does <premise> entails <hypothesis> ? {self.mask_token}"]

	def convert_for_mlm(self, premise, hypothesis, relationship):
		"""
		relationship defined in 0, 1, 2 
		"""
		verbalizer = random.choice(self.verbalizer_pool[relationship])
		template = random.choice(self.template_pool)
		# force string to be treated as raw string to avoid regular expression errors
		# premise = repr(premise)
		# hypothesis = repr(hypothesis)
		# text = re.sub("<premise>", r"{}".format(premise), template)
		# text = re.sub("<hypothesis>", r"{}".format(hypothesis), text)
		# text = re.sub(f"{self.mask_token}", verbalizer, text)
		text = template.replace("<premise>", premise)
		text = text.replace("<hypothesis>", hypothesis)
		text = text.replace("<mask>", verbalizer)
		return text


