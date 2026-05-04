"""Step 3: Template-based verbalization of hyperedges ($0, no LLM)."""

import random


class MedicalTemplateEngine:

    SINGLE = {
        'disease_phenotype_positive': [
            "{anchor} characteristically presents with {nb}",
            "{anchor} is clinically associated with {nb}",
            "Patients with {anchor} typically exhibit {nb}",
            "Clinical features of {anchor} include {nb}",
        ],
        'disease_phenotype_negative': [
            "{nb} are typically ABSENT in {anchor}",
            "{anchor} does NOT present with {nb}",
            "The absence of {nb} helps distinguish {anchor}",
        ],
        'indication': [
            "{anchor} is treated with {nb}",
            "{nb} are indicated for {anchor}",
            "Approved therapies for {anchor} include {nb}",
        ],
        'contraindication': [
            "{nb} are CONTRAINDICATED in {anchor}",
            "Patients with {anchor} should avoid {nb}",
            "{nb} must NOT be used in {anchor}",
        ],
        'off-label use': [
            "{nb} have off-label applications in {anchor}",
        ],
        'drug_protein': [
            "{anchor} targets {nb}",
            "The molecular targets of {anchor} include {nb}",
        ],
        'drug_effect': [
            "Known effects of {anchor} include {nb}",
            "{anchor} produces {nb}",
        ],
        'disease_protein': [
            "Molecular basis of {anchor} involves {nb}",
            "{nb} are associated with {anchor}",
        ],
        'disease_disease': [
            "{anchor} commonly co-occurs with {nb}",
            "{nb} are frequent comorbidities of {anchor}",
        ],
        'exposure_disease': [
            "Risk factors for {anchor} include {nb}",
            "Exposure to {nb} increases risk of {anchor}",
        ],
        'pathway_protein': [
            "The {anchor} pathway involves {nb}",
        ],
        'bioprocess_protein': [
            "{anchor} is mediated by {nb}",
        ],
        'molfunc_protein': [
            "{nb} have the molecular function of {anchor}",
        ],
        'cellcomp_protein': [
            "{nb} are localized to {anchor}",
        ],
        'phenotype_protein': [
            "{anchor} is associated with {nb}",
        ],
        'exposure_protein': [
            "{anchor} exposure affects {nb}",
        ],
        'exposure_bioprocess': [
            "{anchor} influences {nb}",
        ],
        'drug_drug': [
            "{anchor} and {nb} share overlapping therapeutic indications",
        ],
    }

    COMPOSITE_CONN = {
        'presents with': ', presenting with ',
        'notably without': '. Notably absent: ',
        'treated with': '. Treatment includes ',
        'contraindicated with': '. Contraindicated: ',
        'risk factors include': '. Risk factors: ',
        'co-occurs with': '. Often co-occurs with ',
        'indicated for': ' is indicated for ',
        'targets': ', targeting ',
        'effects include': '. Effects include ',
        'contraindicated in': '. Contraindicated in ',
    }

    PATH_T = {
        'symptom_disease_drug': [
            "{e0} suggests {e1}, which is treated with {e2}",
            "The symptom {e0} is a feature of {e1}; therapy includes {e2}",
        ],
        'disease_protein_drug': [
            "{e0} involves {e1}, which is targeted by {e2}",
            "{e2} treats {e0} by acting on {e1}",
        ],
        'drug_protein_pathway': [
            "{e0} targets {e1} in the {e2} pathway",
            "{e0} modulates {e2} through its action on {e1}",
        ],
        'exposure_disease_phenotype': [
            "Exposure to {e0} increases risk of {e1}, manifesting as {e2}",
            "{e0} is a risk factor for {e1}, which presents with {e2}",
        ],
        'comorbidity_drug': [
            "{e0} often co-occurs with {e1}; {e2} may be used for treatment",
        ],
    }

    def verbalize(self, he: dict) -> str:
        t = he['type']
        if t == 'neighbor_agg':
            return self._neighbor(he)
        if t == 'composite':
            return self._composite(he)
        if t == 'path':
            return self._path(he)
        if t == 'feature':
            # Description already fully verbalized by feature_hedges.py; pass through.
            return he['description']
        return str(he)

    # Anchor-type-specific overrides: (relation, anchor_type) -> templates
    # Used when the same relation has different semantics depending on anchor direction.
    SINGLE_BY_ANCHOR: dict[tuple[str, str], list[str]] = {
        # indication: anchor=disease → nb=drugs
        ('indication', 'disease'): [
            "{anchor} is treated with {nb}",
            "{nb} are indicated for {anchor}",
            "Approved therapies for {anchor} include {nb}",
        ],
        # indication: anchor=drug → nb=diseases
        ('indication', 'drug'): [
            "{anchor} is indicated for {nb}",
            "{anchor} is approved for the treatment of {nb}",
        ],
        # contraindication: anchor=disease → nb=drugs
        ('contraindication', 'disease'): [
            "{nb} are CONTRAINDICATED in {anchor}",
            "Patients with {anchor} should avoid {nb}",
            "{nb} must NOT be used in {anchor}",
        ],
        # contraindication: anchor=drug → nb=diseases
        ('contraindication', 'drug'): [
            "{anchor} is contraindicated in patients with {nb}",
            "{anchor} should NOT be used in {nb}",
        ],
        # off-label use: anchor=drug → nb=diseases
        ('off-label use', 'drug'): [
            "{anchor} has off-label applications in {nb}",
            "{anchor} is used off-label for {nb}",
        ],
        # cellcomp_protein: anchor=cellular_component → nb=proteins
        ('cellcomp_protein', 'cellular_component'): [
            "{nb} are localized to {anchor}",
            "{nb} are components of the {anchor}",
        ],
        # exposure_disease: anchor=exposure → nb=diseases
        ('exposure_disease', 'exposure'): [
            "{anchor} exposure increases risk of {nb}",
            "Exposure to {anchor} is associated with {nb}",
        ],
        # exposure_disease: anchor=disease → nb=exposures (default SINGLE is correct)
    }

    def _neighbor(self, he: dict) -> str:
        key = (he['relation'], he.get('anchor_type', ''))
        tpls = self.SINGLE_BY_ANCHOR.get(key) or self.SINGLE.get(he['relation'])
        if not tpls:
            return f"{he['anchor']} is related to {self._lst(he['neighbors'])}"
        return random.choice(tpls).format(
            anchor=he['anchor'], nb=self._lst(he['neighbors'])
        )

    def _composite(self, c: dict) -> str:
        out = [c['anchor']]
        for key, items in c['parts']:
            out.append(self.COMPOSITE_CONN[key] + self._lst(items))
        return ''.join(out) + '.'

    def _path(self, p: dict) -> str:
        tpls = self.PATH_T.get(p['path_pattern'])
        if not tpls:
            e, r = p['entities'], p['relations']
            return f"{e[0]} ({r[0]}) {e[1]} ({r[1]}) {e[2]}"
        e = p['entities']
        return random.choice(tpls).format(e0=e[0], e1=e[1], e2=e[2])

    @staticmethod
    def _lst(items: list[str]) -> str:
        if len(items) == 1:
            return items[0]
        if len(items) == 2:
            return f"{items[0]} and {items[1]}"
        return ', '.join(items[:-1]) + f', and {items[-1]}'
