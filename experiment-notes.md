Experiment 17

Seems to have liked this group a lot:

[logs](http://localhost:6006/#scalars&regexInput=logs_17.(100%7C99%7C96%7C86%7C97%7C80%7C93%7C95%7C87%7C94%7C84%7C65%7C76%7C54%7C75%7C63%7C39%7C83%7C88%7C58%7C41%7C64%7C43%7C45%7C53%7C74%7C35%7C36%7C44%7C85%7C37%7C98%7C78%7C73%7C30%7C)%24)

`logs_17.(100|99|96|86|97|80|93|95|87|94|84|65|76|54|75|63|39|83|88|58|41|64|43|45|53|74|35|36|44|85|37|98|78|73|30|)$`

top ones all have an r of 8, varied at the bottom of the pack
all have a lora alpha of 128, except for the bottom one, which is 64, and the cutoff, which is 256
all have an infonce of 0.0625... So I need to explore lower
all have a batch size of 1
most all have a lora dropout of 0.0
all are 4 bits
adam epsilon is generally towards 1e-10
adam beta2 is around .995-.999
adam beta1 is around .9


For some reason, it doesn't seem to care about this much more promising looking path that has a completely different shape.
Here, comparing it to the best of the other batch

[promising](localhost:6006/#scalars&regexInput=logs_17.(100|89))

Value is : 0.00399074195932425
```
r_exp 3
adam_beta1 0.9
adam_beta2 0.9974
adam_epsilon 1.8434218591156536e-10
lora_dropout 0.0
infonce_temp_exp 1
lora_alpha_exp 8
batch_size_exp 2
bits 4
```

```
batch_size 4
change_in_loss_per_second 0.00399074195932425
duration_seconds 76.40183281898499
final_loss 0.3951
infonce_temp 0.0625
initial_loss 0.5698
lora_alpha 256
r 8
version_trial 17.89
```