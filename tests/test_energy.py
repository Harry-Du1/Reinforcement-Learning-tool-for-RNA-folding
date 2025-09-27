from rl_essential.energy import TurnerEnergyModel

def test_hairpin():
    E = TurnerEnergyModel()
    seq = "GGGAAACCC"
    pairing = [8,-1,-1,-1,-1,-1,-1,-1,0]
    assert E.total_energy(seq, pairing) > -10

def test_stack():
    E = TurnerEnergyModel()
    seq = "GCGC"
    pairing = [3,2,1,0]
    e = E.total_energy(seq, pairing)
    assert e < 0