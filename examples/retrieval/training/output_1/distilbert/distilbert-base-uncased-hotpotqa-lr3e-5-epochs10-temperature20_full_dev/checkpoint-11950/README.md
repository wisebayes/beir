---
tags:
- sentence-transformers
- sentence-similarity
- feature-extraction
- generated_from_trainer
- dataset_size:153000
- loss:MultipleNegativesRankingLoss
base_model: distilbert/distilbert-base-uncased
widget:
- source_sentence: The low cost airline known as Skymark Airlines is headquartered
    at what airport in Japan?
  sentences:
  - 'Tokyo International Airport (東京国際空港 , Tōkyō Kokusai Kūkō ) , commonly known as
    Haneda Airport (羽田空港 , Haneda Kūkō ) or Tokyo Haneda Airport (IATA: HND, ICAO:
    RJTT) , is one of the two primary airports that serve the Greater Tokyo Area,
    and is the primary base of Japan''s two major domestic airlines, Japan Airlines
    (Terminal 1) and All Nippon Airways (Terminal 2), as well as Air Do, Skymark Airlines,
    Solaseed Air, and StarFlyer. It is located in Ōta, Tokyo, 14 km south of Tokyo
    Station.'
  - Major-General Dr. Walter Robert Dornberger (6 September 1895 – 27 June 1980) was
    a German Army artillery officer whose career spanned World War I and World War
    II. He was a leader of Nazi Germany's V-2 rocket program and other projects at
    the Peenemünde Army Research Center.
  - Edgerton "Ed" Hartwell II (born May 27, 1978) is a former American football linebacker.
    He played college football at Western Illinois and was drafted by the Baltimore
    Ravens in the fourth round of the 2001 NFL Draft. Hartwell was also a member of
    the Atlanta Falcons, Cincinnati Bengals, Oakland Raiders and Las Vegas Locomotives.
- source_sentence: Who has been in more bands, Philip Labonte or Dave Williams?
  sentences:
  - Jedd Hughes (born in Quorn, Australia) is a singer, songwriter, session guitar
    player, and record producer.
  - Moves was a wargaming magazine originally published by SPI (Simulations Publications,
    Inc.), who also published manual wargames. Their flagship magazine "Strategy &
    Tactics" ("S&T"), was a military history magazine featuring a new wargame in each
    issue. While S&T was devoted to historical articles, "Moves" focused on the play
    of the games. Each issue carried articles dealing with strategies for different
    wargames, tactical tips, and many variants and scenarios for existing games. As
    time passed, reviews of new games also became an important feature. While the
    majority of the articles dealt with SPI games, the magazine was open to and published
    many articles on games by other companies.
  - Philip Steven Labonte (Born April 15, 1975) is an American musician from Massachusetts,
    best known as the lead singer of the American heavy metal band All That Remains.
    Labonte is the former lead vocalist for Shadows Fall, was the touring vocalist
    for Killswitch Engage in early 2010, and also filled in for Five Finger Death
    Punch vocalist Ivan L. Moody in late 2016.
- source_sentence: The University of Science and Arts of Oklahoma is funded by what
    system?
  sentences:
  - George Jasper Caster (August 4, 1907 – December 18, 1955), nicknamed "Ug", was
    a right-handed professional baseball pitcher for 21 years from 1929 to 1948 and
    again in 1953. He played 12 years in Major League Baseball with the Philadelphia
    Athletics (1934–1935, 1937–1940), St. Louis Browns (1941–1945), and Detroit Tigers
    (1945–1946).
  - The University of Science and Arts of Oklahoma, or USAO, is a public liberal arts
    college located in Chickasha, Oklahoma. It is the only public college in Oklahoma
    with a strictly liberal arts-focused curriculum and is a member of the Council
    of Public Liberal Arts Colleges. USAO is an undergraduate-only institution and
    grants Bachelor's Degrees in a variety of subject areas. The school was founded
    in 1908 as a school for women and from 1912 to 1965 was known as Oklahoma College
    for Women. It became coeducational in 1965 and today educates approximately 1,000
    students. In 2001, the entire Oklahoma College for Women campus was listed as
    a National Historic District.
  - '"Lisa the Greek" is the fourteenth episode of "The Simpsons"<nowiki>''</nowiki>
    third season. It originally aired on the Fox network in the United States on January
    23, 1992. In the episode, Homer begins to bond with his daughter, Lisa, after
    learning her unique and convenient ability to pick winning football teams, but,
    secretly, uses her ability to help him gamble. When Lisa finds out Homer''s secret,
    she refuses to speak to her father until he fully understands her. "Lisa the Greek"
    was written by Jay Kogen and Wallace Wolodarsky, and directed by Rich Moore.'
- source_sentence: The stepchild of Shirley Jones turned down a song later recorded
    by what band?
  sentences:
  - David Bruce Cassidy (born April 12, 1950) is a retired American actor, singer,
    songwriter, and guitarist. He is known for his role as Keith Partridge, the son
    of Shirley Partridge (played by his stepmother Shirley Jones), in the 1970s musical-sitcom
    "The Partridge Family", which led to his becoming one of popular culture's teen
    idols and pop singers of the 1970s. He later had a career in both acting and music.
  - Hal Hartley (born November 3, 1959) is an American film director, screenwriter,
    producer and composer who became a key figure in the American independent film
    movement of the 1980s and '90s. He is best known for his films "Trust", "Amateur"
    and "Henry Fool", which are notable for deadpan humour and offbeat characters
    quoting philosophical dialogue.
  - Str8 Outta Northcote is the second full-length album from Australian grindcore
    band Blood Duster, its title being a parody of the N.W.A album "Straight Outta
    Compton". Northcote is the suburb in Melbourne where most of the band lived at
    the time.
- source_sentence: Which district of Lancashire, England contains the village that
    lies between Langho and a village that is on the banks of the River Calder?
  sentences:
  - James Lawrence (16 February 1879 – 21 November 1934) was a Scottish football player
    and manager. A goalkeeper, he played for Newcastle United between 1904 and 1922.
  - Billington is a village in the Ribble Valley district of Lancashire, England.
    It lies between the villages of Whalley and Langho. It forms part of the Billington
    and Langho civil parish and contains the schools St Augustine's RC High School,
    St Leonard's Primary and St Mary's Primary.
  - The Orange Tundra is a cocktail of Vodka, Kahlúa, Creme Soda, and Orange juice.
    It is traditionally poured over 2 to 3 cubes of ice in an old-fashioned 8-12 oz.
    glass.
pipeline_tag: sentence-similarity
library_name: sentence-transformers
---

# SentenceTransformer based on distilbert/distilbert-base-uncased

This is a [sentence-transformers](https://www.SBERT.net) model finetuned from [distilbert/distilbert-base-uncased](https://huggingface.co/distilbert/distilbert-base-uncased). It maps sentences & paragraphs to a 768-dimensional dense vector space and can be used for semantic textual similarity, semantic search, paraphrase mining, text classification, clustering, and more.

## Model Details

### Model Description
- **Model Type:** Sentence Transformer
- **Base model:** [distilbert/distilbert-base-uncased](https://huggingface.co/distilbert/distilbert-base-uncased) <!-- at revision 12040accade4e8a0f71eabdb258fecc2e7e948be -->
- **Maximum Sequence Length:** 512 tokens
- **Output Dimensionality:** 768 dimensions
- **Similarity Function:** Energy Distance
<!-- - **Training Dataset:** Unknown -->
<!-- - **Language:** Unknown -->
<!-- - **License:** Unknown -->

### Model Sources

- **Documentation:** [Sentence Transformers Documentation](https://sbert.net)
- **Repository:** [Sentence Transformers on GitHub](https://github.com/UKPLab/sentence-transformers)
- **Hugging Face:** [Sentence Transformers on Hugging Face](https://huggingface.co/models?library=sentence-transformers)

### Full Model Architecture

```
SentenceTransformer(
  (0): Transformer({'max_seq_length': 512, 'do_lower_case': False}) with Transformer model: DistilBertModel 
  (1): Pooling({'word_embedding_dimension': 768, 'pooling_mode_cls_token': False, 'pooling_mode_mean_tokens': True, 'pooling_mode_max_tokens': False, 'pooling_mode_mean_sqrt_len_tokens': False, 'pooling_mode_weightedmean_tokens': False, 'pooling_mode_lasttoken': False, 'include_prompt': True})
)
```

## Usage

### Direct Usage (Sentence Transformers)

First install the Sentence Transformers library:

```bash
pip install -U sentence-transformers
```

Then you can load this model and run inference.
```python
from sentence_transformers import SentenceTransformer

# Download from the 🤗 Hub
model = SentenceTransformer("sentence_transformers_model_id")
# Run inference
sentences = [
    'Which district of Lancashire, England contains the village that lies between Langho and a village that is on the banks of the River Calder?',
    "Billington is a village in the Ribble Valley district of Lancashire, England. It lies between the villages of Whalley and Langho. It forms part of the Billington and Langho civil parish and contains the schools St Augustine's RC High School, St Leonard's Primary and St Mary's Primary.",
    'James Lawrence (16 February 1879 – 21 November 1934) was a Scottish football player and manager. A goalkeeper, he played for Newcastle United between 1904 and 1922.',
]
embeddings = model.encode(sentences)
print(embeddings.shape)
# [3, 768]

# Get the similarity scores for the embeddings
similarities = model.similarity(embeddings, embeddings)
print(similarities.shape)
# [3, 3]
```

<!--
### Direct Usage (Transformers)

<details><summary>Click to see the direct usage in Transformers</summary>

</details>
-->

<!--
### Downstream Usage (Sentence Transformers)

You can finetune this model on your own dataset.

<details><summary>Click to expand</summary>

</details>
-->

<!--
### Out-of-Scope Use

*List how the model may foreseeably be misused and address what users ought not to do with the model.*
-->

<!--
## Bias, Risks and Limitations

*What are the known or foreseeable issues stemming from this model? You could also flag here known failure cases or weaknesses of the model.*
-->

<!--
### Recommendations

*What are recommendations with respect to the foreseeable issues? For example, filtering explicit content.*
-->

## Training Details

### Training Dataset

#### Unnamed Dataset

* Size: 153,000 training samples
* Columns: <code>query</code> and <code>text</code>
* Approximate statistics based on the first 1000 samples:
  |         | query                                                                              | text                                                                                |
  |:--------|:-----------------------------------------------------------------------------------|:------------------------------------------------------------------------------------|
  | type    | string                                                                             | string                                                                              |
  | details | <ul><li>min: 7 tokens</li><li>mean: 24.16 tokens</li><li>max: 117 tokens</li></ul> | <ul><li>min: 16 tokens</li><li>mean: 95.71 tokens</li><li>max: 302 tokens</li></ul> |
* Samples:
  | query                                                                                                                                                          | text                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                   |
  |:---------------------------------------------------------------------------------------------------------------------------------------------------------------|:---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
  | <code>The surf reef break which Joel Parkinson surfed on in the ASP World Championship Tour Surfing Title on 14 December 2012 is located in what state?</code> | <code>Joel Parkinson (born April 10, 1981) is an Australian surfer who competes on the ASP (Association Of Surfing Professionals) World Tour. After twelve years competing at the elite level on the ASP World Championship Tour, a stretch that saw him win eleven elite ASP World Title Events, plus nine additional ASP tour events, and achieve runner-up second place to the ASP World Title four times, Parkinson won the ASP World Championship Tour Surfing Title on 14 December 2012 in Hawaii at the Banzai Pipeline during the ASP World Tours' final event for 2012–the Billabong Pipeline Masters. Parkinson hung on in a back and forth battle with eleven-time ASP World Title holder, Kelly Slater, to get his first World Title, as well as go on to win the Pipeline Masters, only after Slater lost his semi-final heat to Josh Kerr, of Queensland, Australia. Parkinson beat Kerr in the finals of the event, which was his seventh top-five placing for the year, and his first event title win for 2012.</code> |
  | <code>What type of publications are both Woman's World and Pictorial Review?</code>                                                                            | <code>Woman's World is an American supermarket weekly magazine with a circulation of 1.6 million readers. Printed on paper generally associated with tabloid publications and priced accordingly, it concentrates on short articles about subjects such as weight loss, relationship advice and cooking, along with feature stories about women in the STEM fields and academia. It has held the title of the most popular newsstand women's magazine, with sales of 77 million copies in 2004. It competes with more general-market traditional magazines such as "Woman's Day" and "Family Circle".</code>                                                                                                                                                                                                                                                                                                                                                                                                                           |
  | <code>Anna Cruz plays basketball for the WNBA team located in what city?</code>                                                                                | <code>Anna Cruz Lebrato (born October 27, 1986) is a Spanish professional basketball player with the Minnesota Lynx of the WNBA. She plays for Nadezhda Orenburg, Minnesota Lynx and Spain women's national basketball team. She has represented national team since 2009, earning bronze medals at EuroBasket Women 2009, and World Championship 2010 and silver medal at World Championship 2014.</code>                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                             |
* Loss: [<code>MultipleNegativesRankingLoss</code>](https://sbert.net/docs/package_reference/sentence_transformer/losses.html#multiplenegativesrankingloss) with these parameters:
  ```json
  {
      "scale": 20.0,
      "similarity_fct": "<lambda>"
  }
  ```

### Evaluation Dataset

#### Unnamed Dataset

* Size: 17,000 evaluation samples
* Columns: <code>query</code> and <code>text</code>
* Approximate statistics based on the first 1000 samples:
  |         | query                                                                              | text                                                                                |
  |:--------|:-----------------------------------------------------------------------------------|:------------------------------------------------------------------------------------|
  | type    | string                                                                             | string                                                                              |
  | details | <ul><li>min: 8 tokens</li><li>mean: 24.62 tokens</li><li>max: 129 tokens</li></ul> | <ul><li>min: 15 tokens</li><li>mean: 95.83 tokens</li><li>max: 387 tokens</li></ul> |
* Samples:
  | query                                                                                                                                                                                            | text                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                     |
  |:-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|:---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
  | <code>Were A Funny Thing Happened on the Way to the Moon and Air Force, Incorporated directed by the same person?</code>                                                                         | <code>A Funny Thing Happened on the Way to the Moon is a 2001 film written, produced and directed by Nashville-based filmmaker Bart Sibrel. Sibrel is a critic of the Apollo program and proponent of the conspiracy theory that the six Apollo Moon landing missions between 1969 and 1972 were elaborate hoaxes perpetrated by the United States government, including NASA.</code>                                                                                                                                                                                                                                                                    |
  | <code>Which gold mine, Camlaren Mine or Burwash Mine is located farther north of Yellowknife?</code>                                                                                             | <code>The Burwash Mine was a small gold property discovered in the fall of 1934 by Johnny Baker and Hugh Muir at Yellowknife Bay, Northwest Territories. The town of Yellowknife did not exist yet at that point, but the discovery of gold at Burwash was the catalyst that brought more gold prospectors into the region in 1935 and 1936. A short shaft was sunk in 1935-1936 at Burwash, and in the summer of 1935 a 16-ton bulk sample of ore was shipped to Trail, British Columbia for processing, yielding 200 troy ounces (6 kg) of gold. The mine did not become a substantial producer and it is believed the gold vein was mined out.</code> |
  | <code>The MKT Nature and Fitness Trail runs nine miles in the right of way of what former Class 1 railroad company that was established in 1865, and had its last headquarters in Dallas?</code> | <code>The Missouri–Kansas–Texas Railroad (reporting mark MKT) is a former Class I railroad company in the United States, with its last headquarters in Dallas. Established in 1865 under the name Union Pacific Railway, Southern Branch, it came to serve an extensive rail network in Texas, Oklahoma, Kansas, and Missouri. In 1988, it merged with the Missouri Pacific Railroad and is now part of Union Pacific Railroad.</code>                                                                                                                                                                                                                   |
* Loss: [<code>MultipleNegativesRankingLoss</code>](https://sbert.net/docs/package_reference/sentence_transformer/losses.html#multiplenegativesrankingloss) with these parameters:
  ```json
  {
      "scale": 20.0,
      "similarity_fct": "<lambda>"
  }
  ```

### Training Hyperparameters
#### Non-Default Hyperparameters

- `eval_strategy`: epoch
- `per_device_train_batch_size`: 16
- `per_device_eval_batch_size`: 16
- `gradient_accumulation_steps`: 4
- `learning_rate`: 3e-05
- `weight_decay`: 0.01
- `num_train_epochs`: 10
- `warmup_steps`: 956
- `seed`: 24
- `fp16`: True
- `ddp_find_unused_parameters`: False

#### All Hyperparameters
<details><summary>Click to expand</summary>

- `overwrite_output_dir`: False
- `do_predict`: False
- `eval_strategy`: epoch
- `prediction_loss_only`: True
- `per_device_train_batch_size`: 16
- `per_device_eval_batch_size`: 16
- `per_gpu_train_batch_size`: None
- `per_gpu_eval_batch_size`: None
- `gradient_accumulation_steps`: 4
- `eval_accumulation_steps`: None
- `torch_empty_cache_steps`: None
- `learning_rate`: 3e-05
- `weight_decay`: 0.01
- `adam_beta1`: 0.9
- `adam_beta2`: 0.999
- `adam_epsilon`: 1e-08
- `max_grad_norm`: 1.0
- `num_train_epochs`: 10
- `max_steps`: -1
- `lr_scheduler_type`: linear
- `lr_scheduler_kwargs`: {}
- `warmup_ratio`: 0.0
- `warmup_steps`: 956
- `log_level`: passive
- `log_level_replica`: warning
- `log_on_each_node`: True
- `logging_nan_inf_filter`: True
- `save_safetensors`: True
- `save_on_each_node`: False
- `save_only_model`: False
- `restore_callback_states_from_checkpoint`: False
- `no_cuda`: False
- `use_cpu`: False
- `use_mps_device`: False
- `seed`: 24
- `data_seed`: None
- `jit_mode_eval`: False
- `use_ipex`: False
- `bf16`: False
- `fp16`: True
- `fp16_opt_level`: O1
- `half_precision_backend`: auto
- `bf16_full_eval`: False
- `fp16_full_eval`: False
- `tf32`: None
- `local_rank`: 0
- `ddp_backend`: None
- `tpu_num_cores`: None
- `tpu_metrics_debug`: False
- `debug`: []
- `dataloader_drop_last`: True
- `dataloader_num_workers`: 0
- `dataloader_prefetch_factor`: None
- `past_index`: -1
- `disable_tqdm`: False
- `remove_unused_columns`: True
- `label_names`: None
- `load_best_model_at_end`: False
- `ignore_data_skip`: False
- `fsdp`: []
- `fsdp_min_num_params`: 0
- `fsdp_config`: {'min_num_params': 0, 'xla': False, 'xla_fsdp_v2': False, 'xla_fsdp_grad_ckpt': False}
- `tp_size`: 0
- `fsdp_transformer_layer_cls_to_wrap`: None
- `accelerator_config`: {'split_batches': False, 'dispatch_batches': None, 'even_batches': True, 'use_seedable_sampler': True, 'non_blocking': False, 'gradient_accumulation_kwargs': None}
- `deepspeed`: None
- `label_smoothing_factor`: 0.0
- `optim`: adamw_torch
- `optim_args`: None
- `adafactor`: False
- `group_by_length`: False
- `length_column_name`: length
- `ddp_find_unused_parameters`: False
- `ddp_bucket_cap_mb`: None
- `ddp_broadcast_buffers`: False
- `dataloader_pin_memory`: True
- `dataloader_persistent_workers`: False
- `skip_memory_metrics`: True
- `use_legacy_prediction_loop`: False
- `push_to_hub`: False
- `resume_from_checkpoint`: None
- `hub_model_id`: None
- `hub_strategy`: every_save
- `hub_private_repo`: None
- `hub_always_push`: False
- `gradient_checkpointing`: False
- `gradient_checkpointing_kwargs`: None
- `include_inputs_for_metrics`: False
- `include_for_metrics`: []
- `eval_do_concat_batches`: True
- `fp16_backend`: auto
- `push_to_hub_model_id`: None
- `push_to_hub_organization`: None
- `mp_parameters`: 
- `auto_find_batch_size`: False
- `full_determinism`: False
- `torchdynamo`: None
- `ray_scope`: last
- `ddp_timeout`: 1800
- `torch_compile`: False
- `torch_compile_backend`: None
- `torch_compile_mode`: None
- `include_tokens_per_second`: False
- `include_num_input_tokens_seen`: False
- `neftune_noise_alpha`: None
- `optim_target_modules`: None
- `batch_eval_metrics`: False
- `eval_on_start`: False
- `use_liger_kernel`: False
- `eval_use_gather_object`: False
- `average_tokens_across_devices`: False
- `prompts`: None
- `batch_sampler`: batch_sampler
- `multi_dataset_batch_sampler`: proportional

</details>

### Training Logs
<details><summary>Click to expand</summary>

| Epoch  | Step  | Training Loss | Validation Loss |
|:------:|:-----:|:-------------:|:---------------:|
| 0.0209 | 25    | 7287.0037     | -               |
| 0.0418 | 50    | 6615.4131     | -               |
| 0.0627 | 75    | 5165.5938     | -               |
| 0.0837 | 100   | 3156.7575     | -               |
| 0.1046 | 125   | 1752.6778     | -               |
| 0.1255 | 150   | 922.3782      | -               |
| 0.1464 | 175   | 468.4045      | -               |
| 0.1673 | 200   | 252.5346      | -               |
| 0.1882 | 225   | 144.4355      | -               |
| 0.2092 | 250   | 83.1875       | -               |
| 0.2301 | 275   | 55.4107       | -               |
| 0.2510 | 300   | 42.8861       | -               |
| 0.2719 | 325   | 34.3134       | -               |
| 0.2928 | 350   | 30.5689       | -               |
| 0.3137 | 375   | 26.081        | -               |
| 0.3347 | 400   | 22.7558       | -               |
| 0.3556 | 425   | 20.0955       | -               |
| 0.3765 | 450   | 18.8336       | -               |
| 0.3974 | 475   | 17.2085       | -               |
| 0.4183 | 500   | 15.4589       | -               |
| 0.4392 | 525   | 14.1082       | -               |
| 0.4602 | 550   | 13.1408       | -               |
| 0.4811 | 575   | 12.5381       | -               |
| 0.5020 | 600   | 12.1148       | -               |
| 0.5229 | 625   | 11.9643       | -               |
| 0.5438 | 650   | 11.4346       | -               |
| 0.5647 | 675   | 11.036        | -               |
| 0.5857 | 700   | 10.6892       | -               |
| 0.6066 | 725   | 10.5678       | -               |
| 0.6275 | 750   | 10.1855       | -               |
| 0.6484 | 775   | 9.7644        | -               |
| 0.6693 | 800   | 9.2872        | -               |
| 0.6902 | 825   | 9.0353        | -               |
| 0.7111 | 850   | 8.4803        | -               |
| 0.7321 | 875   | 10.9496       | -               |
| 0.7530 | 900   | 7.9815        | -               |
| 0.7739 | 925   | 7.7468        | -               |
| 0.7948 | 950   | 7.3679        | -               |
| 0.8157 | 975   | 7.0003        | -               |
| 0.8366 | 1000  | 6.9098        | -               |
| 0.8576 | 1025  | 6.9099        | -               |
| 0.8785 | 1050  | 6.4205        | -               |
| 0.8994 | 1075  | 6.5306        | -               |
| 0.9203 | 1100  | 6.0135        | -               |
| 0.9412 | 1125  | 6.285         | -               |
| 0.9621 | 1150  | 6.0358        | -               |
| 0.9831 | 1175  | 5.5468        | -               |
| 1.0    | 1196  | -             | 1.1511          |
| 1.0033 | 1200  | 5.3224        | -               |
| 1.0243 | 1225  | 5.2385        | -               |
| 1.0452 | 1250  | 5.2977        | -               |
| 1.0661 | 1275  | 4.9996        | -               |
| 1.0870 | 1300  | 5.0568        | -               |
| 1.1079 | 1325  | 4.8159        | -               |
| 1.1288 | 1350  | 4.978         | -               |
| 1.1498 | 1375  | 4.647         | -               |
| 1.1707 | 1400  | 4.5914        | -               |
| 1.1916 | 1425  | 4.7713        | -               |
| 1.2125 | 1450  | 4.5983        | -               |
| 1.2334 | 1475  | 4.4464        | -               |
| 1.2543 | 1500  | 4.5525        | -               |
| 1.2753 | 1525  | 4.443         | -               |
| 1.2962 | 1550  | 4.4114        | -               |
| 1.3171 | 1575  | 4.0618        | -               |
| 1.3380 | 1600  | 4.0393        | -               |
| 1.3589 | 1625  | 4.0492        | -               |
| 1.3798 | 1650  | 4.1664        | -               |
| 1.4008 | 1675  | 4.2287        | -               |
| 1.4217 | 1700  | 4.1111        | -               |
| 1.4426 | 1725  | 3.7994        | -               |
| 1.4635 | 1750  | 3.8316        | -               |
| 1.4844 | 1775  | 3.6228        | -               |
| 1.5053 | 1800  | 3.9557        | -               |
| 1.5262 | 1825  | 3.964         | -               |
| 1.5472 | 1850  | 3.8689        | -               |
| 1.5681 | 1875  | 4.2717        | -               |
| 1.5890 | 1900  | 3.8881        | -               |
| 1.6099 | 1925  | 3.6009        | -               |
| 1.6308 | 1950  | 3.6595        | -               |
| 1.6517 | 1975  | 4.7243        | -               |
| 1.6727 | 2000  | 3.6909        | -               |
| 1.6936 | 2025  | 3.5145        | -               |
| 1.7145 | 2050  | 3.6053        | -               |
| 1.7354 | 2075  | 3.9313        | -               |
| 1.7563 | 2100  | 3.5291        | -               |
| 1.7772 | 2125  | 3.5007        | -               |
| 1.7982 | 2150  | 3.564         | -               |
| 1.8191 | 2175  | 3.4317        | -               |
| 1.8400 | 2200  | 3.5558        | -               |
| 1.8609 | 2225  | 3.3724        | -               |
| 1.8818 | 2250  | 3.5086        | -               |
| 1.9027 | 2275  | 3.498         | -               |
| 1.9237 | 2300  | 3.261         | -               |
| 1.9446 | 2325  | 3.3137        | -               |
| 1.9655 | 2350  | 3.3259        | -               |
| 1.9864 | 2375  | 3.4449        | -               |
| 2.0    | 2392  | -             | 1.0195          |
| 2.0067 | 2400  | 3.0721        | -               |
| 2.0276 | 2425  | 2.9587        | -               |
| 2.0485 | 2450  | 2.9173        | -               |
| 2.0694 | 2475  | 3.0636        | -               |
| 2.0904 | 2500  | 3.0425        | -               |
| 2.1113 | 2525  | 3.3841        | -               |
| 2.1322 | 2550  | 3.0018        | -               |
| 2.1531 | 2575  | 3.1322        | -               |
| 2.1740 | 2600  | 2.8917        | -               |
| 2.1949 | 2625  | 2.8191        | -               |
| 2.2159 | 2650  | 2.9329        | -               |
| 2.2368 | 2675  | 3.3096        | -               |
| 2.2577 | 2700  | 2.7466        | -               |
| 2.2786 | 2725  | 2.8564        | -               |
| 2.2995 | 2750  | 2.705         | -               |
| 2.3204 | 2775  | 2.776         | -               |
| 2.3414 | 2800  | 2.7662        | -               |
| 2.3623 | 2825  | 2.7267        | -               |
| 2.3832 | 2850  | 2.807         | -               |
| 2.4041 | 2875  | 2.6256        | -               |
| 2.4250 | 2900  | 2.8112        | -               |
| 2.4459 | 2925  | 2.7375        | -               |
| 2.4668 | 2950  | 2.7668        | -               |
| 2.4878 | 2975  | 2.6371        | -               |
| 2.5087 | 3000  | 2.8185        | -               |
| 2.5296 | 3025  | 2.7186        | -               |
| 2.5505 | 3050  | 2.7581        | -               |
| 2.5714 | 3075  | 2.638         | -               |
| 2.5923 | 3100  | 2.6773        | -               |
| 2.6133 | 3125  | 2.8809        | -               |
| 2.6342 | 3150  | 2.6301        | -               |
| 2.6551 | 3175  | 2.5227        | -               |
| 2.6760 | 3200  | 2.5075        | -               |
| 2.6969 | 3225  | 2.7651        | -               |
| 2.7178 | 3250  | 2.6704        | -               |
| 2.7388 | 3275  | 2.5489        | -               |
| 2.7597 | 3300  | 2.6637        | -               |
| 2.7806 | 3325  | 2.4705        | -               |
| 2.8015 | 3350  | 2.606         | -               |
| 2.8224 | 3375  | 2.4866        | -               |
| 2.8433 | 3400  | 2.6283        | -               |
| 2.8643 | 3425  | 2.4637        | -               |
| 2.8852 | 3450  | 2.4305        | -               |
| 2.9061 | 3475  | 2.6206        | -               |
| 2.9270 | 3500  | 2.9854        | -               |
| 2.9479 | 3525  | 2.5005        | -               |
| 2.9688 | 3550  | 2.5375        | -               |
| 2.9898 | 3575  | 2.4142        | -               |
| 3.0    | 3588  | -             | 0.7693          |
| 3.0100 | 3600  | 2.1219        | -               |
| 3.0310 | 3625  | 2.1833        | -               |
| 3.0519 | 3650  | 2.3227        | -               |
| 3.0728 | 3675  | 2.3195        | -               |
| 3.0937 | 3700  | 2.3971        | -               |
| 3.1146 | 3725  | 2.2775        | -               |
| 3.1355 | 3750  | 2.2007        | -               |
| 3.1565 | 3775  | 2.0727        | -               |
| 3.1774 | 3800  | 2.1226        | -               |
| 3.1983 | 3825  | 2.2022        | -               |
| 3.2192 | 3850  | 2.3737        | -               |
| 3.2401 | 3875  | 2.4097        | -               |
| 3.2610 | 3900  | 2.3255        | -               |
| 3.2819 | 3925  | 2.3473        | -               |
| 3.3029 | 3950  | 2.2528        | -               |
| 3.3238 | 3975  | 2.2227        | -               |
| 3.3447 | 4000  | 2.3276        | -               |
| 3.3656 | 4025  | 2.1734        | -               |
| 3.3865 | 4050  | 2.1559        | -               |
| 3.4074 | 4075  | 2.1909        | -               |
| 3.4284 | 4100  | 2.1331        | -               |
| 3.4493 | 4125  | 2.1528        | -               |
| 3.4702 | 4150  | 2.0085        | -               |
| 3.4911 | 4175  | 2.1103        | -               |
| 3.5120 | 4200  | 2.1811        | -               |
| 3.5329 | 4225  | 2.0621        | -               |
| 3.5539 | 4250  | 2.1706        | -               |
| 3.5748 | 4275  | 2.1279        | -               |
| 3.5957 | 4300  | 2.0157        | -               |
| 3.6166 | 4325  | 2.0705        | -               |
| 3.6375 | 4350  | 2.0638        | -               |
| 3.6584 | 4375  | 2.0952        | -               |
| 3.6794 | 4400  | 2.2106        | -               |
| 3.7003 | 4425  | 2.0929        | -               |
| 3.7212 | 4450  | 2.1225        | -               |
| 3.7421 | 4475  | 2.01          | -               |
| 3.7630 | 4500  | 1.9513        | -               |
| 3.7839 | 4525  | 2.0939        | -               |
| 3.8049 | 4550  | 2.1791        | -               |
| 3.8258 | 4575  | 2.0137        | -               |
| 3.8467 | 4600  | 1.9789        | -               |
| 3.8676 | 4625  | 1.9638        | -               |
| 3.8885 | 4650  | 1.9747        | -               |
| 3.9094 | 4675  | 2.0563        | -               |
| 3.9303 | 4700  | 1.99          | -               |
| 3.9513 | 4725  | 2.1325        | -               |
| 3.9722 | 4750  | 2.0793        | -               |
| 3.9931 | 4775  | 1.9708        | -               |
| 4.0    | 4784  | -             | 0.6912          |
| 4.0134 | 4800  | 1.8964        | -               |
| 4.0343 | 4825  | 1.7733        | -               |
| 4.0552 | 4850  | 2.0228        | -               |
| 4.0761 | 4875  | 1.8613        | -               |
| 4.0971 | 4900  | 1.9426        | -               |
| 4.1180 | 4925  | 1.8675        | -               |
| 4.1389 | 4950  | 1.7717        | -               |
| 4.1598 | 4975  | 1.9079        | -               |
| 4.1807 | 5000  | 1.9474        | -               |
| 4.2016 | 5025  | 1.6996        | -               |
| 4.2225 | 5050  | 1.8244        | -               |
| 4.2435 | 5075  | 1.8897        | -               |
| 4.2644 | 5100  | 1.8312        | -               |
| 4.2853 | 5125  | 1.784         | -               |
| 4.3062 | 5150  | 1.9437        | -               |
| 4.3271 | 5175  | 1.9276        | -               |
| 4.3480 | 5200  | 2.2377        | -               |
| 4.3690 | 5225  | 1.9203        | -               |
| 4.3899 | 5250  | 1.7298        | -               |
| 4.4108 | 5275  | 1.8156        | -               |
| 4.4317 | 5300  | 1.7559        | -               |
| 4.4526 | 5325  | 1.9351        | -               |
| 4.4735 | 5350  | 1.7261        | -               |
| 4.4945 | 5375  | 1.6836        | -               |
| 4.5154 | 5400  | 1.8694        | -               |
| 4.5363 | 5425  | 1.8241        | -               |
| 4.5572 | 5450  | 1.9027        | -               |
| 4.5781 | 5475  | 1.7204        | -               |
| 4.5990 | 5500  | 1.7944        | -               |
| 4.6200 | 5525  | 1.7942        | -               |
| 4.6409 | 5550  | 1.746         | -               |
| 4.6618 | 5575  | 1.7907        | -               |
| 4.6827 | 5600  | 1.7963        | -               |
| 4.7036 | 5625  | 1.5814        | -               |
| 4.7245 | 5650  | 1.8949        | -               |
| 4.7455 | 5675  | 1.7438        | -               |
| 4.7664 | 5700  | 1.7925        | -               |
| 4.7873 | 5725  | 1.8806        | -               |
| 4.8082 | 5750  | 1.7472        | -               |
| 4.8291 | 5775  | 1.6966        | -               |
| 4.8500 | 5800  | 1.7364        | -               |
| 4.8709 | 5825  | 1.7778        | -               |
| 4.8919 | 5850  | 1.7092        | -               |
| 4.9128 | 5875  | 1.6966        | -               |
| 4.9337 | 5900  | 1.7479        | -               |
| 4.9546 | 5925  | 1.7206        | -               |
| 4.9755 | 5950  | 3.0665        | -               |
| 4.9964 | 5975  | 1.7979        | -               |
| 5.0    | 5980  | -             | 0.5596          |
| 5.0167 | 6000  | 1.5502        | -               |
| 5.0376 | 6025  | 1.6591        | -               |
| 5.0586 | 6050  | 1.5392        | -               |
| 5.0795 | 6075  | 1.6809        | -               |
| 5.1004 | 6100  | 1.7549        | -               |
| 5.1213 | 6125  | 1.6717        | -               |
| 5.1422 | 6150  | 1.5319        | -               |
| 5.1631 | 6175  | 1.5763        | -               |
| 5.1841 | 6200  | 1.6248        | -               |
| 5.2050 | 6225  | 1.624         | -               |
| 5.2259 | 6250  | 1.793         | -               |
| 5.2468 | 6275  | 1.5836        | -               |
| 5.2677 | 6300  | 1.6908        | -               |
| 5.2886 | 6325  | 1.6538        | -               |
| 5.3096 | 6350  | 1.6085        | -               |
| 5.3305 | 6375  | 1.5635        | -               |
| 5.3514 | 6400  | 1.6836        | -               |
| 5.3723 | 6425  | 1.4843        | -               |
| 5.3932 | 6450  | 1.576         | -               |
| 5.4141 | 6475  | 1.5629        | -               |
| 5.4351 | 6500  | 1.5441        | -               |
| 5.4560 | 6525  | 1.5024        | -               |
| 5.4769 | 6550  | 1.6217        | -               |
| 5.4978 | 6575  | 1.5497        | -               |
| 5.5187 | 6600  | 1.5781        | -               |
| 5.5396 | 6625  | 1.691         | -               |
| 5.5606 | 6650  | 1.6745        | -               |
| 5.5815 | 6675  | 1.4831        | -               |
| 5.6024 | 6700  | 1.4973        | -               |
| 5.6233 | 6725  | 1.4531        | -               |
| 5.6442 | 6750  | 1.4839        | -               |
| 5.6651 | 6775  | 1.6858        | -               |
| 5.6860 | 6800  | 1.5338        | -               |
| 5.7070 | 6825  | 1.4586        | -               |
| 5.7279 | 6850  | 1.4829        | -               |
| 5.7488 | 6875  | 1.6737        | -               |
| 5.7697 | 6900  | 1.5708        | -               |
| 5.7906 | 6925  | 1.605         | -               |
| 5.8115 | 6950  | 1.6209        | -               |
| 5.8325 | 6975  | 1.7195        | -               |
| 5.8534 | 7000  | 1.5269        | -               |
| 5.8743 | 7025  | 1.5767        | -               |
| 5.8952 | 7050  | 1.7042        | -               |
| 5.9161 | 7075  | 1.4981        | -               |
| 5.9370 | 7100  | 1.5898        | -               |
| 5.9580 | 7125  | 1.5837        | -               |
| 5.9789 | 7150  | 1.6348        | -               |
| 5.9998 | 7175  | 1.4435        | -               |
| 6.0    | 7176  | -             | 0.5450          |
| 6.0201 | 7200  | 1.303         | -               |
| 6.0410 | 7225  | 1.3333        | -               |
| 6.0619 | 7250  | 1.4236        | -               |
| 6.0828 | 7275  | 1.355         | -               |
| 6.1037 | 7300  | 1.4204        | -               |
| 6.1247 | 7325  | 1.4935        | -               |
| 6.1456 | 7350  | 1.4483        | -               |
| 6.1665 | 7375  | 1.4814        | -               |
| 6.1874 | 7400  | 1.4663        | -               |
| 6.2083 | 7425  | 1.423         | -               |
| 6.2292 | 7450  | 1.4879        | -               |
| 6.2502 | 7475  | 1.4656        | -               |
| 6.2711 | 7500  | 1.5062        | -               |
| 6.2920 | 7525  | 1.3841        | -               |
| 6.3129 | 7550  | 1.4056        | -               |
| 6.3338 | 7575  | 1.4394        | -               |
| 6.3547 | 7600  | 1.3588        | -               |
| 6.3757 | 7625  | 1.3317        | -               |
| 6.3966 | 7650  | 1.2351        | -               |
| 6.4175 | 7675  | 1.4927        | -               |
| 6.4384 | 7700  | 1.4045        | -               |
| 6.4593 | 7725  | 2.1578        | -               |
| 6.4802 | 7750  | 1.3068        | -               |
| 6.5012 | 7775  | 1.4378        | -               |
| 6.5221 | 7800  | 1.4498        | -               |
| 6.5430 | 7825  | 1.4703        | -               |
| 6.5639 | 7850  | 1.4237        | -               |
| 6.5848 | 7875  | 1.405         | -               |
| 6.6057 | 7900  | 1.4271        | -               |
| 6.6266 | 7925  | 1.4564        | -               |
| 6.6476 | 7950  | 1.3717        | -               |
| 6.6685 | 7975  | 1.3911        | -               |
| 6.6894 | 8000  | 1.4441        | -               |
| 6.7103 | 8025  | 1.5035        | -               |
| 6.7312 | 8050  | 1.3734        | -               |
| 6.7521 | 8075  | 1.3301        | -               |
| 6.7731 | 8100  | 1.4058        | -               |
| 6.7940 | 8125  | 1.4477        | -               |
| 6.8149 | 8150  | 1.4175        | -               |
| 6.8358 | 8175  | 1.3346        | -               |
| 6.8567 | 8200  | 1.4478        | -               |
| 6.8776 | 8225  | 1.6391        | -               |
| 6.8986 | 8250  | 1.3905        | -               |
| 6.9195 | 8275  | 1.3644        | -               |
| 6.9404 | 8300  | 1.4779        | -               |
| 6.9613 | 8325  | 1.2928        | -               |
| 6.9822 | 8350  | 1.3961        | -               |
| 7.0    | 8372  | -             | 0.5206          |
| 7.0025 | 8375  | 1.4219        | -               |
| 7.0234 | 8400  | 1.3406        | -               |
| 7.0443 | 8425  | 1.3372        | -               |
| 7.0653 | 8450  | 1.4108        | -               |
| 7.0862 | 8475  | 1.2952        | -               |
| 7.1071 | 8500  | 1.2675        | -               |
| 7.1280 | 8525  | 1.3569        | -               |
| 7.1489 | 8550  | 1.3471        | -               |
| 7.1698 | 8575  | 1.4294        | -               |
| 7.1908 | 8600  | 1.2304        | -               |
| 7.2117 | 8625  | 1.4733        | -               |
| 7.2326 | 8650  | 1.2949        | -               |
| 7.2535 | 8675  | 1.248         | -               |
| 7.2744 | 8700  | 1.3229        | -               |
| 7.2953 | 8725  | 1.2472        | -               |
| 7.3163 | 8750  | 1.1973        | -               |
| 7.3372 | 8775  | 1.3863        | -               |
| 7.3581 | 8800  | 1.3483        | -               |
| 7.3790 | 8825  | 1.2368        | -               |
| 7.3999 | 8850  | 1.3537        | -               |
| 7.4208 | 8875  | 1.3836        | -               |
| 7.4417 | 8900  | 1.2606        | -               |
| 7.4627 | 8925  | 1.3183        | -               |
| 7.4836 | 8950  | 1.3057        | -               |
| 7.5045 | 8975  | 1.3114        | -               |
| 7.5254 | 9000  | 1.3575        | -               |
| 7.5463 | 9025  | 1.352         | -               |
| 7.5672 | 9050  | 1.3745        | -               |
| 7.5882 | 9075  | 1.2205        | -               |
| 7.6091 | 9100  | 1.1614        | -               |
| 7.6300 | 9125  | 1.2377        | -               |
| 7.6509 | 9150  | 1.2143        | -               |
| 7.6718 | 9175  | 1.2606        | -               |
| 7.6927 | 9200  | 1.2849        | -               |
| 7.7137 | 9225  | 1.3321        | -               |
| 7.7346 | 9250  | 1.2426        | -               |
| 7.7555 | 9275  | 1.2089        | -               |
| 7.7764 | 9300  | 1.2699        | -               |
| 7.7973 | 9325  | 1.3348        | -               |
| 7.8182 | 9350  | 1.3324        | -               |
| 7.8392 | 9375  | 1.2922        | -               |
| 7.8601 | 9400  | 1.3035        | -               |
| 7.8810 | 9425  | 1.257         | -               |
| 7.9019 | 9450  | 1.2461        | -               |
| 7.9228 | 9475  | 1.2263        | -               |
| 7.9437 | 9500  | 1.3314        | -               |
| 7.9647 | 9525  | 1.3288        | -               |
| 7.9856 | 9550  | 1.1715        | -               |
| 8.0    | 9568  | -             | 0.5382          |
| 8.0059 | 9575  | 1.2106        | -               |
| 8.0268 | 9600  | 1.2001        | -               |
| 8.0477 | 9625  | 1.2402        | -               |
| 8.0686 | 9650  | 1.1983        | -               |
| 8.0895 | 9675  | 1.2042        | -               |
| 8.1104 | 9700  | 1.2645        | -               |
| 8.1314 | 9725  | 1.2925        | -               |
| 8.1523 | 9750  | 1.2022        | -               |
| 8.1732 | 9775  | 1.2146        | -               |
| 8.1941 | 9800  | 1.1749        | -               |
| 8.2150 | 9825  | 1.2332        | -               |
| 8.2359 | 9850  | 1.1626        | -               |
| 8.2569 | 9875  | 1.285         | -               |
| 8.2778 | 9900  | 1.241         | -               |
| 8.2987 | 9925  | 1.2578        | -               |
| 8.3196 | 9950  | 1.149         | -               |
| 8.3405 | 9975  | 1.2628        | -               |
| 8.3614 | 10000 | 1.1719        | -               |
| 8.3823 | 10025 | 1.185         | -               |
| 8.4033 | 10050 | 1.387         | -               |
| 8.4242 | 10075 | 1.219         | -               |
| 8.4451 | 10100 | 1.1838        | -               |
| 8.4660 | 10125 | 1.1868        | -               |
| 8.4869 | 10150 | 1.2127        | -               |
| 8.5078 | 10175 | 1.179         | -               |
| 8.5288 | 10200 | 1.2175        | -               |
| 8.5497 | 10225 | 1.1798        | -               |
| 8.5706 | 10250 | 1.1369        | -               |
| 8.5915 | 10275 | 1.1484        | -               |
| 8.6124 | 10300 | 1.1654        | -               |
| 8.6333 | 10325 | 1.2833        | -               |
| 8.6543 | 10350 | 1.1613        | -               |
| 8.6752 | 10375 | 1.2383        | -               |
| 8.6961 | 10400 | 1.2603        | -               |
| 8.7170 | 10425 | 1.1515        | -               |
| 8.7379 | 10450 | 1.1914        | -               |
| 8.7588 | 10475 | 1.1099        | -               |
| 8.7798 | 10500 | 1.2897        | -               |
| 8.8007 | 10525 | 1.1796        | -               |
| 8.8216 | 10550 | 1.1895        | -               |
| 8.8425 | 10575 | 1.289         | -               |
| 8.8634 | 10600 | 1.2408        | -               |
| 8.8843 | 10625 | 1.1896        | -               |
| 8.9052 | 10650 | 1.0744        | -               |
| 8.9262 | 10675 | 1.0684        | -               |
| 8.9471 | 10700 | 1.2395        | -               |
| 8.9680 | 10725 | 1.2127        | -               |
| 8.9889 | 10750 | 1.0345        | -               |
| 9.0    | 10764 | -             | 0.5190          |
| 9.0092 | 10775 | 1.1541        | -               |
| 9.0301 | 10800 | 1.1123        | -               |
| 9.0510 | 10825 | 1.1156        | -               |
| 9.0720 | 10850 | 1.0912        | -               |
| 9.0929 | 10875 | 1.1304        | -               |
| 9.1138 | 10900 | 1.0935        | -               |
| 9.1347 | 10925 | 1.225         | -               |
| 9.1556 | 10950 | 1.1124        | -               |
| 9.1765 | 10975 | 1.1434        | -               |
| 9.1974 | 11000 | 1.1377        | -               |
| 9.2184 | 11025 | 1.1538        | -               |
| 9.2393 | 11050 | 1.1947        | -               |
| 9.2602 | 11075 | 1.0736        | -               |
| 9.2811 | 11100 | 1.0518        | -               |
| 9.3020 | 11125 | 1.2051        | -               |
| 9.3229 | 11150 | 1.0998        | -               |
| 9.3439 | 11175 | 1.0105        | -               |
| 9.3648 | 11200 | 1.1695        | -               |
| 9.3857 | 11225 | 1.0959        | -               |
| 9.4066 | 11250 | 1.1288        | -               |
| 9.4275 | 11275 | 1.1414        | -               |
| 9.4484 | 11300 | 1.0788        | -               |
| 9.4694 | 11325 | 1.1237        | -               |
| 9.4903 | 11350 | 1.153         | -               |
| 9.5112 | 11375 | 1.1356        | -               |
| 9.5321 | 11400 | 1.1615        | -               |
| 9.5530 | 11425 | 1.2092        | -               |
| 9.5739 | 11450 | 1.1708        | -               |
| 9.5949 | 11475 | 1.1514        | -               |
| 9.6158 | 11500 | 1.109         | -               |
| 9.6367 | 11525 | 1.0681        | -               |
| 9.6576 | 11550 | 1.1138        | -               |
| 9.6785 | 11575 | 1.0752        | -               |
| 9.6994 | 11600 | 1.1278        | -               |
| 9.7204 | 11625 | 1.1797        | -               |
| 9.7413 | 11650 | 1.1509        | -               |
| 9.7622 | 11675 | 1.1377        | -               |
| 9.7831 | 11700 | 1.1384        | -               |
| 9.8040 | 11725 | 1.1375        | -               |
| 9.8249 | 11750 | 1.176         | -               |
| 9.8458 | 11775 | 1.1155        | -               |
| 9.8668 | 11800 | 1.1114        | -               |
| 9.8877 | 11825 | 1.0621        | -               |
| 9.9086 | 11850 | 1.0567        | -               |
| 9.9295 | 11875 | 1.1742        | -               |
| 9.9504 | 11900 | 1.1123        | -               |
| 9.9713 | 11925 | 1.0973        | -               |
| 9.9923 | 11950 | 1.1446        | -               |

</details>

### Framework Versions
- Python: 3.9.21
- Sentence Transformers: 3.5.0.dev0
- Transformers: 4.51.3
- PyTorch: 2.1.2+cu121
- Accelerate: 1.6.0
- Datasets: 2.21.0
- Tokenizers: 0.21.1

## Citation

### BibTeX

#### Sentence Transformers
```bibtex
@inproceedings{reimers-2019-sentence-bert,
    title = "Sentence-BERT: Sentence Embeddings using Siamese BERT-Networks",
    author = "Reimers, Nils and Gurevych, Iryna",
    booktitle = "Proceedings of the 2019 Conference on Empirical Methods in Natural Language Processing",
    month = "11",
    year = "2019",
    publisher = "Association for Computational Linguistics",
    url = "https://arxiv.org/abs/1908.10084",
}
```

#### MultipleNegativesRankingLoss
```bibtex
@misc{henderson2017efficient,
    title={Efficient Natural Language Response Suggestion for Smart Reply},
    author={Matthew Henderson and Rami Al-Rfou and Brian Strope and Yun-hsuan Sung and Laszlo Lukacs and Ruiqi Guo and Sanjiv Kumar and Balint Miklos and Ray Kurzweil},
    year={2017},
    eprint={1705.00652},
    archivePrefix={arXiv},
    primaryClass={cs.CL}
}
```

<!--
## Glossary

*Clearly define terms in order to be accessible across audiences.*
-->

<!--
## Model Card Authors

*Lists the people who create the model card, providing recognition and accountability for the detailed work that goes into its construction.*
-->

<!--
## Model Card Contact

*Provides a way for people who have updates to the Model Card, suggestions, or questions, to contact the Model Card authors.*
-->