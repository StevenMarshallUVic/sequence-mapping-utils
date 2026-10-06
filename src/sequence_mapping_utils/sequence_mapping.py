import logging
import os
from dataclasses import dataclass
from pathlib import Path
from multiprocessing import current_process
from typing import Self, Iterable

from Bio import SeqIO
from Bio.Seq import Seq, MutableSeq
from Bio.SeqRecord import SeqRecord

logger = logging.getLogger(Path(__file__).name)


@dataclass(frozen=True)
class SequenceMapper:
    """Provides helper methods for mapping sequences to IDs.

    Attributes
    ----------
    seq_to_ids
        Mapping between sequences and their corresponding IDs.
    """

    seq_to_ids: dict[Seq, tuple[str, ...]]

    def find_ids_for_sequences(
            self,
            sequences: list[Seq],
            log_bin_size: int = 25
    ) -> dict[Seq, tuple[str, ...] | None]:
        """Find IDs for each sequence.

        Parameters
        ----------
        sequences
            Sequences to find IDs for.
        log_bin_size
            How many sequences to process between each status debug log message.
            If number of sequences is less than this value, status logs are
            not written.

        Returns
        -------
        dict[Seq, tuple[str, ...] | None]
            Mapping between sequences and their corresponding IDs,
            or None if no matching sequence could be found.
        """

        process_name = current_process().name
        process_id = os.getpid()
        log_status = len(sequences) > log_bin_size

        seq_to_id: dict[Seq, tuple[str, ...] | None] = {}
        for i, seq in enumerate(sequences):
            if log_status and (i % log_bin_size == 0):
                logger.debug(
                    f"{process_name} - {process_id}: "
                    f"{i}/{len(sequences)}...",
                )
            seq_to_id[seq] = self.find_id_for_sequence(seq)

        if log_status:
            logger.debug(f"{process_name} - {process_id}: Done!")

        return seq_to_id

    def find_id_for_sequence(
            self,
            sequence: Seq,
    ) -> tuple[str, ...] | None:
        """Find ID for a sequence.

        Parameters
        ----------
        sequence
            Sequence to find ID for.

        Returns
        -------
        tuple[str, ...] | None
            ID match, or None if no matching sequence could be found.
        """
        return self._find_id_for_sequence(sequence)

    def _find_id_for_sequence(
            self,
            sequence: Seq,
            trim_count=0
    ) -> tuple[str, ...] | None:
        """Find ID for a sequence.

        Parameters
        ----------
        sequence
            Sequence to find ID for.
        trim_count
            How many residues to trim from the termini of the sequence.

        Returns
        -------
        tuple[str, ...] | None
            ID match, or None if no matching sequence could be found.
        """

        trimmed_sequence = Seq(str(sequence)[trim_count:-trim_count]) \
            if trim_count > 0 else sequence

        # Attempt to find direct match
        if trimmed_sequence in self.seq_to_ids:
            return self.seq_to_ids[trimmed_sequence]

        # Attempt to find subset match
        ids = {
            self.seq_to_ids[seq]
            for seq in self.seq_to_ids
            if trimmed_sequence in seq
        }
        if len(ids) == 1:
            return ids.pop()
        if len(ids) > 1:
            raise ValueError(
                f"Multiple subset matches for '{trimmed_sequence}'. "
                f"Matches: '{ids}'. Full sequence: '{sequence}'."
            )

        # Attempt to find match with sequence trimmed if still long enough
        if len(trimmed_sequence) > 8:
            return self._find_id_for_sequence(
                sequence,
                trim_count=trim_count + 1
            )

        return None

    @classmethod
    def from_records(
            cls,
            records: Iterable[SeqRecord],
            id_split: str | None = None,
    ) -> Self:
        """Create instance by mapping records to their IDs.

        Parameters
        ----------
        records
            Records to map sequences to IDs for.
        id_split
            Optional substring to split upon for each sequence's ID.
        """

        seq_to_id: dict[Seq, tuple[str, ...]] = {}
        for record in records:
            seq = record.seq
            if seq is None or isinstance(seq, MutableSeq):
                raise ValueError(f"Expected type `Seq`, got `{type(seq)}`.")

            record_id: str | None = record.id
            if record_id is None:
                raise ValueError(f"Invalid id for record: {record}")

            record_id: str = record_id if id_split is None \
                else record_id.split(id_split)[0]

            seq_to_id[seq] = (*seq_to_id.get(seq, tuple()), record_id)

        return cls(
            seq_to_ids=seq_to_id,
        )

    @classmethod
    def from_fasta(cls, fasta: Path, *args, **kwargs) -> Self:
        """Create instance by mapping sequences in a FASTA file to their IDs.

        See `from_records` method for descriptions of additional parameters.

        Parameters
        ----------
        fasta
            FASTA to map sequences to IDs for.
        """

        return cls.from_records(
            SeqIO.parse(fasta, "fasta"),
            *args,
            **kwargs
        )
