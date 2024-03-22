from mgz.typing import *


class DocumentRuleEvaluator:
    def __init__(self,
                 rule_set: Union[str, Tuple, Tuple[str, str, str, int]]):
        self.rule_set = rule_set

    def evaluate_rule(self, rules: Union[
        str, Tuple, Tuple[str, int], Tuple[str, str, str, int]],
                      doc: str):
        # 6 scenarios
        # An "or" with other rules, no score at the end
        # An "or" with only words, has score at the end
        # An "and" with other rules, has score at the end
        # An "and" with only words, has score at the end
        # A single word with a score, would be child of an "or"
        # A single word without score, would be child of an "and"
        if isinstance(rules, str):
            if rules in doc:
                return 1
            else:
                return 0
        if rules[0] in ("or", "and"):
            if isinstance(rules[-1], int):
                score: int = rules[-1]
                tokens_to_check = rules[1:-1]
                true_val = False
                if rules[0] == "or":
                    true_val = any([word in doc for word in tokens_to_check])
                elif rules[0] == "and":
                    vals = [word in doc for word in tokens_to_check]
                    true_val = all(vals)
                if true_val:
                    return score
                else:
                    return 0
            else:
                if rules[0] == "or":
                    highest_true_score = max(
                        self.evaluate_rule(rule, doc) for rule in rules[1:])
                    if highest_true_score == 0:
                        return 0
                    else:
                        return highest_true_score
                else:
                    raise ValueError(f"No score found for 'and' rule {rules}")
        else:
            assert len(rules) == 2
            word_to_match, score = rules
            return score if word_to_match in doc else 0

    def check_document(self, document):
        return self.evaluate_rule(self.rule_set, document)


def main():
    rule_set = ("or",
                ("inquir", 1),
                ("investigat", 1),
                ("government", 1),
                ("FERC", 2),
                ("and", "government", "investigat", 3),
                ("and", "government", "inquir", 3),
                ("and", "FERC", "investigat", 4),
                ("and", "FERC", "inquir", 4),
                ("and", "FERC", "government", 4),
                )

    document = """
    The FERC's investigating enron for market manipulation. The FERC investigation primarily focused on Enron's role in the California energy crisis of 2000-2001, along with its trading practices and their impact on electricity markets across the United States. Determine if the email should be produced as evidence based on the document request.
    """
    match = DocumentRuleEvaluator(rule_set)
    print(match.check_document(document))


if __name__ == "__main__":
    main()
