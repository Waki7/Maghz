
# class BeamInference2:
#     def __init__(self, n_beams: int, eos_token_id: int, max_seq_len: int,
#                  length_penalty=1):
#         self.n_beams: int = n_beams
#         self._kill_val = -1e9
#         self.best_probs: List[FloatTensorT['B,NBeams']] = []
#         self.pred_tokens: List[LongTensorT['B,NBeams']] = []
#         self.beam_indices: List[LongTensorT['B,NBeams']] = []
#
#         self.eos_token_id: int = eos_token_id
#         self.max_seq_len: int = max_seq_len
#
#         # these will be initialized for every new inference
#         self.done_beams: List[List[int]] = []  # batch x beam size
#         self.beam_end: List[List[int]] = []  # batch x beam size
#         self.lowest_done_beam_score: FloatTensorT['NBeams'] = \
#             torch.ones(self.n_beams).to(settings.DEVICE) * torch.nan
#
#         self.is_done = False
#         self.length_penalty = length_penalty
#
#     def _all_beams_finished(self):
#         return all(
#             [len(beams) == self.n_beams for beams in self.done_beams])
#
#     def select_ids_from_logprobs(self, log_probs: FloatTensorT[
#         'NBeams*B,VocabSize']) -> LongTensorT['NBeams*B,1']:
#         vocab_size = log_probs.shape[-1]
#         batch_size = (int)(log_probs.size(0) // self.n_beams)
#         seq_len = len(self.best_probs)
#         if len(self.done_beams) == 0:
#             [self.done_beams.append([]) for _ in range(0, batch_size)]
#             [self.beam_end.append([]) for _ in range(0, batch_size)]
#             self.lowest_done_beam_score = torch.ones(batch_size).to(
#                 settings.DEVICE) * torch.nan
#
#         # since it's a probability conditional on the previous ones, and we
#         # took the log prob, we can just add the previous probabilities
#         if seq_len > 0:
#             log_probs += self.best_probs[-1].view(
#                 batch_size * self.n_beams).unsqueeze(-1).expand_as(log_probs)
#
#         log_probs: FloatTensorT['B,NBeams,VocabSize'] = \
#             log_probs.view(batch_size, self.n_beams, vocab_size)
#
#         # if first step then we want to avoid resampling the same tokens from different beams, so set to neg inf
#         if seq_len == 0:
#             log_probs[:, 1:, :] = self._kill_val
#         #
#         # # set beams that already predicted an eos token to -1e9
#         # for batch_i, completed_beam_batch in enumerate(
#         #         self.done_beams):
#         #     for completed_beam in completed_beam_batch:
#         #         log_probs[batch_i, completed_beam, :] = self._kill_val
#         # log_probs[
#         #     batch_i, completed_beam, self.eos_token_id] = -1 * self._kill_val
#
#         log_probs = log_probs.reshape(-1,
#                                       self.n_beams * vocab_size)
#         topk_probs, topk_indices = \
#             torch.topk(log_probs, k=2 * self.n_beams, dim=-1, largest=True,
#                        sorted=True)
#
#         topk_probs: FloatTensorT['B,NBeams'] = topk_probs
#         topk_indices: LongTensorT['B,NBeams'] = topk_indices
#         vocab_idx = topk_indices % vocab_size
#         beam_idx = topk_indices // vocab_size
#
#         # add beams that are done (predict eos) basically pruning the beam
#         done_indices: LongTensorT['N,NDim'] = torch.argwhere(
#             vocab_idx == self.eos_token_id)
#         for batch, new_i in done_indices:
#             from_beam = beam_idx[batch][new_i]
#             topk_probs[batch][new_i] = self._kill_val
#             if len(self.done_beams[batch]) < self.n_beams:
#                 self.done_beams[batch].append(from_beam)
#                 # since haven't added to best_probs yet, todo so not sure on -1
#                 self.beam_end[batch].append(seq_len - 1)
#                 self.lowest_done_beam_score[batch] = min(
#                     self.lowest_done_beam_score[batch],
#                     topk_probs[batch, from_beam] / (
#                             seq_len ** self.length_penalty))
#         # decide if done
#         self.is_done = self._all_beams_finished() or (
#                 torch.max(topk_probs, dim=1)[
#                     0] < self.lowest_done_beam_score).all() or \
#                        seq_len == self.max_seq_len
#
#         # we took extra beams earlier in case some would complete, we only want to keep exploring incomplete beams
#         _, beams_to_continue = \
#             torch.topk(topk_probs, k=self.n_beams, dim=1, largest=True,
#                        sorted=True)
#
#         def mps_gather_bug_workaround(_input, _dim, _index):
#             # return torch.gather(_input.unsqueeze(-1), _dim,
#             #                     _index.unsqueeze(-1)).squeeze(-1)
#             return torch.gather(_input, _dim, _index)
#
#         topk_probs = mps_gather_bug_workaround(topk_probs, -1,
#                                                beams_to_continue)
#         vocab_idx = mps_gather_bug_workaround(vocab_idx, -1, beams_to_continue)
#         beam_idx = mps_gather_bug_workaround(beam_idx, -1, beams_to_continue)
#         self.best_probs.append(topk_probs)
#         self.pred_tokens.append(vocab_idx)
#         self.beam_indices.append(beam_idx)
#
#         # view 1 at the end so that it can be concatenated into sequence
#         return vocab_idx.flatten()
#
#     # TODO: is this finding the max prob redundant? because of log probs summing
#     def get_best_sequence(self) -> List[LongTensorT['TgtSeqLen']]:
#         def backtrack_beam(batch_idx, beam_num, beam_end):
#             tokens: List[int] = []
#             next_beam_num = beam_num
#             for i in range(beam_end, -1, -1):
#                 tokens.insert(0,
#                               self.pred_tokens[i][
#                                   batch_idx, next_beam_num].item())
#                 next_beam_num = self.beam_indices[i][
#                     batch_idx, next_beam_num].item()
#             return torch.LongTensor(tokens)
#
#         sequence_per_batch: List[LongTensorT['TgtSeqLen']] = []
#
#         for batch_idx, batch_beam_num in enumerate(self.done_beams):
#             batch_beam_end = self.beam_end[batch_idx]
#             best_score = float("-inf")
#             best_sequence_for_batch = None
#
#             for beam_num, beam_end in zip(batch_beam_num, batch_beam_end):
#                 score = self.best_probs[beam_end][batch_idx, beam_num] / (
#                         (beam_end + 1) ** self.length_penalty)
#                 if score > best_score:
#                     best_sequence_for_batch = backtrack_beam(batch_idx,
#                                                              beam_num, beam_end)
#                     best_score = score
#             sequence_per_batch.append(best_sequence_for_batch)
#         return sequence_per_batch
