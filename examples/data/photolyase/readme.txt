To support Christou et al. "Time-resolved crystallography captures
light-driven DNA repair", for each of 11 datasets, 5 files are provided:

 -- {timepoint}_deposit.pdb & {timepoint}_deposit.cif 
      Definition of the modeled molecular structure, equivalent 
      information in both PDB and mmCIF format.

 -- {timepoint}_deposit.mtz
      Intensity, structure factor, map data. Columns are 
      as follows:

  * H/K/L - Miller indices 
  * IMEAN/SIGIMEAN - merged intensities from CrystFEL
  * F/SIGF - amplitudes estimated from STARANISO
  * KFEXTR/SIGKFEXTR - extrapolated structure factors from xtrapol8
  * R-free-flags - free flags
  * F-model/PHIF-model - model amplitudes and phases 
  * 2FOFCWT/PH2FOFCWT - density map, with low resolution reflections
      filled with F-model values (from extrapolated refinement). Note an 
      anisotropic cutoff has been applied and treated, high resolution 
      reflections are not filled with computed estimates.
  * 2FOFCWT_no_fill/PH2FOFCWT_no_fill - density map, without low 
      resolution reflections filled (from extrapolated refinement)
  * FOFCWT/PHFOFCWT - difference map (from extrapolated refinement)

 -- D_{id}_val-report-full-P1.pdf
      Validation report provided by the PDB

 -- {timepoint}-dark_kwt_ded.mtz
      Time-resolved difference electron density map coefficients, 
      computed with the k-weighting method reported in the paper.


In addition, provided are:
* restraints files used for refinement 
* difference electron density map coefficients for high pump power vs. 
  low pump power data


