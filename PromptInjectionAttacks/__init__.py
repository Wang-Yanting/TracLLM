
from .DefaultAttacker import DefaultAttacker
from .CombinedAttacker import CombinedAttacker
from .DenyAttacker import DenyAttacker
from .EscapeAttacker import EscapeAttacker
from .FakeAttacker import FakeAttacker
from .IgnoreAttacker import IgnoreAttacker
from .NeuralAttacker import NeuralAttacker
from .CitationAttacker import CitationAttacker

def create_attacker(attack_strategy):
    """
    Factory method to create an attacker
    """
    if attack_strategy == 'combined':
        return CombinedAttacker(attack_strategy)
    if attack_strategy == 'deny':
        return DenyAttacker(attack_strategy)
    if attack_strategy == 'escape':
        return EscapeAttacker(attack_strategy)
    if attack_strategy == 'fake':
        return FakeAttacker(attack_strategy)
    if attack_strategy == 'ignore':
        return IgnoreAttacker(attack_strategy)
    if attack_strategy =="neural":
        return NeuralAttacker(attack_strategy)
    elif attack_strategy == 'default':
        return DefaultAttacker(attack_strategy)
    elif attack_strategy == 'cite':
        return CitationAttacker(attack_strategy)
    err_msg = f"{attack_strategy} is not a valid attack strategy."
    raise ValueError(err_msg)