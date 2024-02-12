"""
A model of sequence evolution on a Russian roulette holey fitness landscape 
to simulate the accumulation of Dobzhansky-Muller incompatibilities (DMIs).
"""

__author__ = 'Ata Kalirad, Ricardo B. R. Azevedo'

__version__ = '1.0'

from copy import *
from itertools import *

import numpy.random as rnd
import pandas as pd


# dict storing viable genotypes during evolution
viability_dict = {}
nu_dict = {}


def int2seq(x, N):
    '''
    Convert integer to sequence of length N.

    Parameters
    ----------
    x : int
        Integer representation of sequence.
    N : int
        Number of loci.

    Returns
    -------
    str
        Sequence.
    '''
    assert x < 2 ** N
    return format(x, 'b').zfill(N)


class Genotype(object):
    """Genotype and corresponding viability.

    Attributes
    ----------
    L : int
        Sequence length.
    p : float
        Probability that a genotype is viable.
    seq : str
        Sequence.
    viable : bool
        Whether genotype is viable.
    """

    def __init__(self, seq, p):
        """Initialize Genotype object from sequence.  

        Parameters
        ----------
        seq : str
            Sequence.
        p : float
            Probability the genotype is viable.
        """
        self.seq = seq
        self.L = len(seq)
        assert p >= 0
        assert p <= 1
        self.p = p
        self.get_viability()

    def get_viability(self):
        """The genotype is viable with probability p and inviable with
        probability 1-p.
        """
        if self.seq in viability_dict:
            self.viable = viability_dict[self.seq]
        else:
            rand = rnd.random()
            if rand < self.p:
                self.viable = True
            else:
                self.viable = False
            viability_dict.update({self.seq: self.viable})

    @staticmethod
    def make_viable(seq, p):
        """Create Genotype object from sequence and make it viable.

        Parameters
        ----------
        seq : str
            Sequence.
        p : float
            Probability that a genotype is viable.

        Returns
        -------
        Genotype
            Viable genotype.
        """
        genotype = Genotype(seq, p)
        if not genotype.viable:
            genotype.viable = True
            viability_dict[seq] = True
        return genotype

    @staticmethod
    def random_sequence(L):
        """Generate a sequence of a certain length.

        Parameters
        ----------
        L : int
            Sequence length.

        Returns
        -------
        Genotype
            Random sequence.
        """
        seq = ''
        rand = rnd.binomial(1, .5, L)
        for i in rand:
            seq += str(i)
        return seq

    @staticmethod
    def random_viable(L, p):
        """Generate viable genotype of a certain length.

        Parameters
        ----------
        L : int
            Sequence length.
        p : float
            Probability the genotype is viable.
        """
        found = False
        while not found:
            seq = Genotype(Genotype.random_sequence(L), p)
            if seq.viable:
                found = True
        return seq

    @staticmethod
    def mutate(seq, site):
        """Mutate sequence at specific site(s).

        Parameters
        ----------
        seq : str
            Sequence.
        site : int or list
            Site(s).

        Returns
        -------
        str
            Mutant sequence

        Examples
        --------
        >>> Genotype.mutate('00000', [0, 1])
        '11000'
        """
        if type(site) is int:
            site = [site]
        for i in site:
            mut = str(abs(int(seq[i]) - 1))
            seq = seq[:i] + mut + seq[i + 1:]
        return seq

    @staticmethod
    def mutate_random(seq):
        """Mutate sequence at a single randomly chosen site.

        Parameters
        ----------
        seq : str
            Sequence.

        Returns
        -------
        str
            Mutant sequence.
        """
        site = rnd.randint(0, len(seq))
        return Genotype.mutate(seq, site)

    @staticmethod
    def dist(seqA, seqB):
        """Calculate Hamming distance between two sequences.

        Parameters
        ----------
        seqA : str
            Sequence.
        seqB : str
            Sequence.

        Returns
        -------
        int
            Hamming distance.

        Examples
        --------
        >>> seqA = '00000'
        >>> seqB = '00110'
        >>> Genotype.dist(seqA, seqB)
        2
        """
        assert type(seqA) == type(seqB)
        diffs = 0
        for ch1, ch2 in zip(seqA, seqB):
            if ch1 != ch2:
                diffs += 1
        return diffs

    @staticmethod
    def get_diverged_sites(seq1, seq2):
        """Find diverged sites between two sequences.

        Parameters
        ----------
        seq1 : str
            Sequence.
        seq2 : str
            Sequence.

        Returns
        -------
        dict
            seq1_alleles: list
                Diverged alleles on seq1.
            seq2_alleles: list
                Diverged alleles on seq2.
            sites: list
                Diverged sites.

        Examples
        --------
        >>> seqA = '000000'
        >>> seqB = '011010'
        >>> Genotype.get_diverged_sites(seqA, seqB)
        {'seq1_alleles': ['0', '0', '0'], 'seq2_alleles': ['1', '1', '1'], 'sites': [1, 2, 4]}
        """
        L = len(seq1)
        assert len(seq2) == L
        seq1_alleles = []
        seq2_alleles = []
        sites = []
        for i in range(L):
            if seq1[i] != seq2[i]:
                seq1_alleles.append(seq1[i])
                seq2_alleles.append(seq2[i])
                sites.append(i)
        return {'seq1_alleles': seq1_alleles, 'seq2_alleles': seq2_alleles, 'sites': sites}

    @staticmethod
    def introgress(recipient, donor):
        """Construct all possible single site introgressions from a donor
        sequence to a recipient sequence.

        Parameters
        ----------
        recipient : str
            Sequence.
        donor : str
            Sequence.

        Returns
        -------
        list
            Introgressed genotypes.

        Examples
        --------
        >>> Genotype.introgress('00000', '01110')
        ['01000', '00100', '00010']
        """
        L = len(recipient)
        assert len(donor) == L
        diverged = Genotype.get_diverged_sites(recipient, donor)
        introgressions = []
        if len(diverged['sites']) > 0:
            for i in diverged['sites']:
                introgressions.append(Genotype.mutate(recipient, i))
        return introgressions

    def get_IIs(self, donor):
        """Find incompatible introgressions (IIs) from donor genotype.

        Parameters
        ----------
        donor : Genotype
            Donor genotype.

        Returns
        -------
        dict
            IIs {site: allele}.
        """
        assert self.p == donor.p
        assert self.L == donor.L
        assert self.viable
        assert donor.viable
        introgressions = Genotype.introgress(self.seq, donor.seq)
        inviable = {}
        for i in introgressions:
            seq = Genotype(i, self.p)
            if not seq.viable:
                diverged = Genotype.get_diverged_sites(self.seq, i)
                inviable.update({diverged['sites'][0]: diverged['seq2_alleles'][0]})
        return inviable

    def get_viable_neighbors(self):
        """Get viable genotypes that are one mutation away from a genotype.

        Parameters
        ----------
        seq : Genotype
            Genotype

        Returns
        -------
        list
            Viable neighbors.
        """
        viable = []
        for i in range(self.L):
            mut = Genotype(Genotype.mutate(self.seq, i), self.p)
            if mut.viable:
                viable.append(i)
        return viable

    def get_robustness(self):
        if self.seq in nu_dict:
            self.nu = nu_dict[self.seq]
        else:
            viable = self.get_viable_neighbors()
            self.nu = len(viable) / self.L
            nu_dict.update({self.seq: self.nu})


class Population(object):

    def __init__(self, ancestor):
        """Initialize Population object from a viable genotype. The population
        is represented by a single genotype.

        Parameters
        ----------
        ancestor : Genotype
            Ancestral genotype.
        """
        assert ancestor.viable
        ancestor.get_robustness()
        self.ancestor = ancestor
        self.current = deepcopy(ancestor)
        self.p = ancestor.p
        self.n_steps = 0
        self.dist = 0
        self.history = {self.n_steps: {'seq': self.current.seq, 'dist': self.dist, 'nu': self.current.nu}}

    def substitute(self, blind):
        '''Attempt substitution.  There are two regimes:

        blind=True: "blind ant" random walk.  One of the genotype's mutational
        neighbors is chosen at random. If the mutant is viable, the population
        moves to the mutant genotype; otherwise, the population
        remains at the current genotype for another time step.

        blind=False: "myopic ant" random walk.  One of the genotype's mutational
        neighbors is chosen at random. If the mutant is viable, the population
        moves to the mutant genotype; otherwise, another mutational
        neighbor is chosen. The process is repeated until a viable mutant is
        found. 

        Parameters
        ----------
        blind : bool
            Whether to conduct a "blind ant" random walk.
        '''
        found = False
        while not found:
            mut = Genotype(Genotype.mutate_random(self.current.seq), self.p)
            if mut.viable:
                found = True
                self.current = mut
            else:
                if blind:
                    found = True
        self.current.get_robustness()
        self.n_steps += 1
        self.dist = Genotype.dist(self.current.seq, self.ancestor.seq)
        self.history.update({self.n_steps: {'seq': self.current.seq, 'dist': self.dist, 'nu': self.current.nu}})


class Orr(object):

    def __init__(self, ancestor):
        '''Found two diverging lineages from a viable genotype.

        Parameters
        ----------
        ancestor : Genotype
            Ancestral genotype.
        '''        
        self.pop1 = Population(ancestor)
        self.pop2 = deepcopy(self.pop1)
        self.dist = 0
        self.II1 = {}
        self.II2 = {}
        self.p = ancestor.p
        self.n_steps = 0
        self.history = {self.n_steps: {'seq1': self.pop1.current.seq, 'seq2': self.pop2.current.seq,
            'dist01': self.pop1.dist, 'dist02': self.pop2.dist,'dist12': self.dist, 
            'II1': self.II1, 'II2': self.II2, 
            'nu1': self.pop1.current.nu, 'nu2': self.pop2.current.nu}}

    def get_dist(self):
        '''Calculate Hamming distance between the two populations.
        '''        
        self.dist = Genotype.dist(self.pop1.current.seq, self.pop2.current.seq)

    def get_IIs(self):
        '''Infer IIs between the two populations (in both directions).
        '''        
        self.II1 = self.pop1.current.get_IIs(self.pop2.current)
        self.II2 = self.pop2.current.get_IIs(self.pop1.current)

    def substitute(self, blind):
        '''Attempt substitution in a randomly chosen population.

        Parameters
        ----------
        blind : bool
            Whether to conduct a "blind ant" random walk.
        '''        
        rand = rnd.random()
        if rand < 0.5:
            self.pop1.substitute(blind)
        else:
            self.pop2.substitute(blind)
        self.n_steps += 1
        self.get_dist()
        self.II1 = {}
        self.II2 = {}
        if self.dist > 1:
            self.get_IIs()
        self.history.update({self.n_steps: {'seq1': self.pop1.current.seq, 'seq2': self.pop2.current.seq,
            'dist01': self.pop1.dist, 'dist02': self.pop2.dist,'dist12': self.dist, 
            'II1': self.II1, 'II2': self.II2, 
            'nu1': self.pop1.current.nu, 'nu2': self.pop2.current.nu}})
        
    def get_stats(self):
        '''Calculate summary statistics from the evolutionary process. 

        Returns
        -------
        pandas DataFrame
            Summary statistics.
        '''        
        seq1 = []
        seq2 = []
        d01 = []
        d02 = []
        d12 = []
        II1 = []
        II2 = []
        nu1 = []
        nu2 = []
        for t in range(len(self.history)):
            seq1.append(self.history[t]['seq1'])
            seq2.append(self.history[t]['seq2'])
            d01.append(self.history[t]['dist01'])
            d02.append(self.history[t]['dist02'])
            d12.append(self.history[t]['dist12'])
            II1.append(len(self.history[t]['II1']))
            II2.append(len(self.history[t]['II2']))
            nu1.append(self.history[t]['nu1'])
            nu2.append(self.history[t]['nu2'])
        data = pd.DataFrame({'seq1': seq1, 'seq2': seq2, 'd01': d01, 'd02': d02, 'd12': d12, 
            'II1': II1, 'II2': II2, 'nu1': nu1, 'nu2': nu2})
        data['p'] = self.p
        data['nu'] = pd.Series(nu_dict).mean()
        return data


if __name__ == "__main__":
    import doctest
    doctest.testmod()
