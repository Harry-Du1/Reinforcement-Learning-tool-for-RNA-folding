from .energy import ToyEnergyModel


def test_energy_unpaired_penalty():
    E = ToyEnergyModel()
    seq = "GCAU" # length 4
    pairing = [-1, -1, -1, -1]
    e = E.total_energy(seq, pairing)
    assert e > 0.0 # all unpaired: positive penalty


def test_energy_pair_bonus():
    E = ToyEnergyModel()
    seq = "GC" # perfect GC pair
    pairing = [1, 0]
    e = E.total_energy(seq, pairing)
    assert e < 0.0 # GC bonus dominates