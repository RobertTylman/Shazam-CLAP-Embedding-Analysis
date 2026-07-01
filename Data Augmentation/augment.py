#!/usr/bin/env python3
"""CLI entry point for audio data augmentation."""

import argparse
import sys
from tqdm import tqdm
from augment_core import NoiseSource, run_augmentation, run_pitch_shift_augmentation


def parse_noise(spec: str) -> NoiseSource:
    """Parse a --noise argument into a NoiseSource."""
    if spec.lower() == "white":
        return NoiseSource(name="white_noise", kind="white")
    if spec.startswith("file:"):
        parts = spec.split(":", 2)
        if len(parts) != 3:
            raise argparse.ArgumentTypeError(
                f"File noise must be file:<name>:<path>, got: {spec}"
            )
        _, name, path = parts
        return NoiseSource(name=name, kind="file", path=path)
    raise argparse.ArgumentTypeError(
        f"Unknown noise spec: {spec}. Use 'white' or 'file:<name>:<path>'"
    )


def main():
    parser = argparse.ArgumentParser(
        description="Augment audio datasets with noise at various SNR levels."
    )
    parser.add_argument("--input", required=True, help="Input audio directory")
    parser.add_argument("--output", required=True, help="Output directory")
    parser.add_argument("--noise", action="append", default=[],
                        help="Noise source: 'white' or 'file:<name>:<path>'. Repeatable.")
    parser.add_argument("--snr", nargs="+", type=float, default=[],
                        help="SNR levels in dB (e.g. 20 10 0)")
    parser.add_argument("--pitch-shift", action="store_true",
                        help="Also generate pitch-shifted copies at +1, +2, and +3 semitones.")
    parser.add_argument("--snippet-duration", type=float, default=30.0,
                        help="Seconds to extract from noise files (default: 30)")
    parser.add_argument("--seed", type=int, default=None,
                        help="Random seed for reproducibility")
    parser.add_argument("--workers", type=int, default=4,
                        help="Number of parallel workers (default: 4)")

    args = parser.parse_args()

    if not args.noise and not args.pitch_shift:
        parser.error("Specify at least one of --noise or --pitch-shift.")
    if args.noise and not args.snr:
        parser.error("--snr is required when --noise is specified.")

    noise_sources = [parse_noise(spec) for spec in args.noise]

    print(f"Input:         {args.input}")
    print(f"Output:        {args.output}")
    if noise_sources:
        print(f"Noise:         {', '.join(ns.name for ns in noise_sources)}")
        print(f"SNR (dB):      {', '.join(str(s) for s in args.snr)}")
    if args.pitch_shift:
        print(f"Pitch shift:   +1, +2, +3 semitones")
    print(f"Workers:       {args.workers}")
    print()

    pbar = tqdm(total=0, unit="file", dynamic_ncols=True)

    def progress(completed, total, last_file):
        if pbar.total != total:
            pbar.total = total
            pbar.refresh()
        pbar.update(1)
        pbar.set_postfix_str(last_file.split("/")[-1], refresh=False)

    total_written = 0

    if noise_sources:
        total_written += run_augmentation(
            input_dir=args.input,
            output_dir=args.output,
            noise_sources=noise_sources,
            snr_levels=args.snr,
            snippet_duration=args.snippet_duration,
            seed=args.seed,
            workers=args.workers,
            progress_callback=progress,
        )

    if args.pitch_shift:
        if noise_sources:
            pbar.close()
            pbar = tqdm(total=0, unit="file", dynamic_ncols=True)
            print("\nRunning pitch-shift augmentation...")
        total_written += run_pitch_shift_augmentation(
            input_dir=args.input,
            output_dir=args.output,
            semitones=[1, 2, 3],
            workers=args.workers,
            progress_callback=progress,
        )

    pbar.close()
    print(f"\nDone. {total_written} files written.")


if __name__ == "__main__":
    main()
