import awkward as ak
from pocket_coffea.lib.cut_definition import Cut

def vbs_ss_dilepton(events, params, year, sample, **kwargs):

    # Masks for VBS Channel with two leptons with the same charge
    mask_2lep = (events.nLeptonGood == 2)

    mask_ss = (events.ll.charge != 0)

    mask_pt_lead = (ak.firsts(events.LeptonGood.pt) > params["pt_leading_lepton"])

    mask_z_veto = (abs(events.ll.mass - 91.2) > params["mll_z_veto"])

    mask_jets = (events.nJetGood >= params["n_jets"])

    mask = ( mask_2lep & mask_ss & mask_pt_lead & mask_z_veto & mask_jets )

    # Pad None values with False
    return ak.where(ak.is_none(mask), False, mask)

vbs_ss_dilepton_presel = Cut(
    name="vbs_ss_dilepton",
    params={
        "pt_leading_lepton": 25,
        "mll_z_veto": 15, 
        "n_jets": 2,     
    },
    function=vbs_ss_dilepton,
)

def nLepton_skim(events, params, **kwargs):
    '''
    Este corte de skim selecciona eventos con al menos 2 leptones
    (muones O electrones) en el evento crudo.
    '''
    nlep = ak.num(events.Muon, axis=1) + ak.num(events.Electron, axis=1)
    return nlep >= 2

nLepton_skim_cut = Cut(name="nLepton_skim", params={}, function=nLepton_skim)
