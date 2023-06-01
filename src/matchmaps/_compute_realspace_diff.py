"""Compute unbiased real space difference map."""

import argparse
import os
import subprocess
import time
from functools import partial

import gemmi
import numpy as np
import reciprocalspaceship as rs

from matchmaps._utils import (
    _handle_special_positions,
    # align_grids_from_model_transform,
    make_floatgrid_from_mtz,
    rigid_body_refinement_wrapper,
    _realspace_align_and_subtract,
    _rbr_selection_parser,
    _renumber_waters,
    # _clean_up_files,
)


def compute_realspace_difference_map(
    pdboff,
    mtzoff,
    mtzon,
    Foff,
    SigFoff,
    Fon,
    SigFon,
    ligands=None,
    dmin=None,
    spacing=0.5,
    on_as_stationary=False,
    input_dir="./",
    output_dir="./",
    verbose=False,
    rbr_selections=None,
    eff=None,
):
    """
    Compute a real-space difference map from mtzs.

    Parameters
    ----------
    pdboff : string
        Name of input .pdb file to use for phasing
    mtzoff : string
        Name of input .mtz file containing 'off' data
    mtzon : string
        Name of input .mtz file containing 'off' data
    Foff : string
        Column in mtzoff containing structure factor amplitudes
    SigFoff : string
        Column in mtzoff containing structure factor uncertainties
    Fon : string
        Column in mtzon containing structure factor amplitudes
    SigFon : string
        Column in mtzon containing structure factor uncertainties
    ligands : list of strings
        Filename(s) of any .cif ligand restraint files necessary for phenix.refine
        by default None, meaning only the .pdb is required for refinement
    dmin : float, optional
        Minimum resolution (in Angstroms) reflections to be used in computing real-space maps from mtzs.
        If omitted, resolution cutoff is the maximum resolution from the lower-resolution input file.
    spacing : float, optional
        Approximate size of real-space voxels in Angstroms, by default 0.5 A
    on_as_stationary : bool, optional
        If True, align "off" data onto "on" data, by default False
    input_dir : str, optional
        Path to directory containing input files, by default "./" (current directory)
    output_dir : str, optional
        Path to directory to which output files should be written, by default "./" (current directory)
    verbose : bool, optional
        If True, print outputs of scaleit and phenix.refine, by default False
    rbr_selections : list of strings, optional
        Custom selections to provide to refinement.refine.sites.rigid_body=
        If omitted, then refinement.refine.sites.rigid_body=all, and the entire structure is refined as a single rigid body.
    eff : str, optional
        Name of a file containing a template .eff parameter file for phenix.refine.
        If omitted, the sensible built-in .eff template is used. If you need to change something,
        I recommend copying the template from the source code and editing that.
    """
    off_name = str(mtzoff.removesuffix(".mtz"))
    on_name = str(mtzon.removesuffix(".mtz"))

    # make sure directories have a trailing slash!
    if input_dir[-1] != "/":
        input_dir = input_dir + "/"

    if output_dir[-1] != "/":
        output_dir = output_dir + "/"

    # take in the list of rbr selections and parse them into phenix and gemmi selection formats
    # if rbr_groups = None, just returns (None, None)
    rbr_phenix, rbr_gemmi = _rbr_selection_parser(rbr_selections)

    # mtzon_scaled = mtzon.removesuffix(".mtz") + "_scaled" + ".mtz"
    # mtzon_scaled = mtzon
    
    # print(
    #     f"{time.strftime('%H:%M:%S')}: Running scaleit to scale 'on' data to 'off' data..."
    # )

    # subprocess.run(
    #     f"rs.scaleit -r {input_dir}/{mtzoff} {Foff} {SigFoff} -i {input_dir}/{mtzon} {Fon} {SigFon} -o {output_dir}/{mtzon_scaled}",
    #     shell=True,
    #     capture_output=(not verbose),
    # )

    pdboff = _handle_special_positions(pdboff, input_dir, output_dir)

    pdboff = _renumber_waters(pdboff, output_dir)

    # mtzon = mtzon_scaled

    print(f"{time.strftime('%H:%M:%S')}: Running phenix.refine for the 'on' data...")

    nickname_on = rigid_body_refinement_wrapper(
        mtzon=mtzon,
        pdboff=pdboff,
        input_dir=input_dir,
        output_dir=output_dir,
        ligands=ligands,
        eff=eff,
        verbose=verbose,
        rbr_selections=rbr_phenix,
        off_labels=f"{Fon},{SigFon}"
    )

    print(f"{time.strftime('%H:%M:%S')}: Running phenix.refine for the 'off' data...")

    nickname_off = rigid_body_refinement_wrapper(
        mtzon=mtzoff,
        pdboff=pdboff,
        input_dir=input_dir,
        output_dir=output_dir,
        ligands=ligands,
        eff=eff,
        verbose=verbose,
        rbr_selections=rbr_phenix,
        off_labels=f"{Foff},{SigFoff}",
    )

    # read back in the files created by phenix
    # these have knowable names
    mtzon = rs.read_mtz(f"{output_dir}/{nickname_on}_1.mtz")
    mtzoff = rs.read_mtz(f"{output_dir}/{nickname_off}_1.mtz")

    pdbon = gemmi.read_structure(f"{output_dir}/{nickname_on}_1.pdb")
    pdboff = gemmi.read_structure(f"{output_dir}/{nickname_off}_1.pdb")

    if dmin is None:
        dmin = max(
            min(mtzoff.compute_dHKL(inplace=True).dHKL),
            min(mtzon.compute_dHKL(inplace=True).dHKL),
        )

    print(f"{time.strftime('%H:%M:%S')}: Constructing FloatGrids from mtzs...")
    # hard-coding F, Phi because they're always phenix outputs
    # TO-DO: Figure out why phenix outputs are sometimes still split into (+) and (-) columns, even when I specify that anomalous=False
    # As a workaround, even anomalous files have a single 'F-obs-filtered' column, so I can always just use that.
    fg_off = make_floatgrid_from_mtz(
        mtzoff, spacing, F="F-obs-filtered", Phi="PH2FOFCWT", spacegroup="P1", dmin=dmin
    )
    fg_on = make_floatgrid_from_mtz(
        mtzon, spacing, F="F-obs-filtered", Phi="PH2FOFCWT", spacegroup="P1", dmin=dmin
    )

    if rbr_gemmi is None:
        _realspace_align_and_subtract(
            fg_off=fg_off,
            fg_on=fg_on,
            pdboff=pdboff,
            pdbon=pdbon,
            output_dir=output_dir,
            on_name=on_name,
            off_name=off_name,
            on_as_stationary=on_as_stationary,
            selection=rbr_gemmi,
        )

    else:  # run helper function separately for each selection
        for n, selection in enumerate(rbr_gemmi, start=1):
            on_name_rbr = on_name + "_rbrgroup" + str(n)
            off_name_rbr = off_name + "_rbrgroup" + str(n)

            _realspace_align_and_subtract(
                fg_off=fg_off.clone(),
                fg_on=fg_on.clone(),
                pdboff=pdboff,
                pdbon=pdbon,
                output_dir=output_dir,
                on_name=on_name_rbr,
                off_name=off_name_rbr,
                on_as_stationary=on_as_stationary,
                selection=selection,
            )
    # print(f"{time.strftime('%H:%M:%S')}: Cleaning up files...")

    # _clean_up_files()

    print(f"{time.strftime('%H:%M:%S')}: Done!")

    return


def parse_arguments():
    """Parse commandline arguments."""
    parser = argparse.ArgumentParser(
        description=(
            "Compute a real-space difference map. "
            "You will need two MTZ files, which will be referred to throughout as 'on' and 'off', "
            "though they could also be light/dark, bound/apo, mutant/WT, hot/cold, etc. "
            "Each mtz will need to contain structure factor amplitudes and uncertainties; you will not need any phases. "
            "You will, however, need an input model (assumed to correspond with the 'off' state) which will be used to determine phases. "
            "Please note that both ccp4 and phenix must be installed and active in your environment for this function to run. "
            ""
            "If you'd like to make an internal difference map instead, see matchmaps.ncs "
        )
    )

    parser.add_argument(
        "--mtzoff",
        "-f",
        nargs=3,
        metavar=("mtzfileoff", "Foff", "SigFoff"),
        required=True,
        help=(
            "MTZ containing off/apo/ground/dark state data. "
            "Specified as [filename F SigF]"
        ),
    )

    parser.add_argument(
        "--mtzon",
        "-n",
        nargs=3,
        metavar=("mtzfileon", "Fon", "SigFon"),
        required=True,
        help=(
            "MTZ containing on/bound/excited/bright state data. "
            "Specified as [filename F SigF]"
        ),
    )

    parser.add_argument(
        "--pdboff",
        "-p",
        required=True,
        help=(
            "Reference pdb corresponding to the off/apo/ground/dark state. "
            "Used for rigid-body refinement of both input MTZs to generate phases."
        ),
    )

    parser.add_argument(
        "--ligands",
        "-l",
        required=False,
        default=None,
        nargs="*",
        help=("Any .cif restraint files needed for refinement"),
    )

    parser.add_argument(
        "--input-dir",
        "-i",
        required=False,
        default="./",
        help="Path to input mtzs and pdb. Optional, defaults to './' (current directory)",
    )

    parser.add_argument(
        "--output-dir",
        "-o",
        required=False,
        default="./",
        help="Path to which output files should be written. Optional, defaults to './' (current directory)",
    )

    parser.add_argument(
        "--on-as-stationary",
        required=False,
        action="store_true",
        default=False,
        help=(
            "Include this flag to align 'off' data onto 'on' data. By default, 'off' data is stationary and 'on' data is moved."
        ),
    )

    parser.add_argument(
        "--spacing",
        "-s",
        required=False,
        type=float,
        default=0.5,
        help=(
            "Approximate voxel size in Angstroms for real-space maps. Defaults to 0.5 A. "
            "Value is approximate because there must be an integer number of voxels along each unit cell dimension"
        ),
    )

    parser.add_argument(
        "--dmin",
        required=False,
        type=float,
        default=None,
        help=(
            "Highest-resolution (in Angstroms) reflections to include in Fourier transform for FloatGrid creation. "
            "By default, cutoff is the resolution limit of the lower-resolution input MTZ. "
        ),
    )

    parser.add_argument(
        "--verbose",
        "-v",
        required=False,
        action="store_true",
        default=False,
        help="Include this flag to print out scaleit and phenix.refine outputs to the terminal. Useful for troubleshooting, but annoying; defaults to False.",
    )

    parser.add_argument(
        "--rbr-selections",
        "-r",
        required=False,
        default=None,
        nargs="*",
        help=(
            "Specification of multiple rigid-body groups for refinement. By default, everything is refined as one rigid-body group. "
        ),
    )

    parser.add_argument(
        "--eff",
        required=False,
        default=None,
        help=("Custom .eff template for running phenix.refine. "),
    )

    return parser


def main():
    parser = parse_arguments()
    args = parser.parse_args()

    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    if not os.path.exists(args.input_dir):
        raise ValueError(f"Input directory '{args.input_dir}' does not exist")

    compute_realspace_difference_map(
        pdboff=args.pdboff,
        ligands=args.ligands,
        mtzoff=args.mtzoff[0],
        mtzon=args.mtzon[0],
        Foff=args.mtzoff[1],
        SigFoff=args.mtzoff[2],
        Fon=args.mtzon[1],
        SigFon=args.mtzon[2],
        input_dir=args.input_dir,
        output_dir=args.output_dir,
        verbose=args.verbose,
        rbr_selections=args.rbr_selections,
        eff=args.eff,
        dmin=args.dmin,
        spacing=args.spacing,
        on_as_stationary=args.on_as_stationary,
    )

    return


if __name__ == "__main__":
    main()
