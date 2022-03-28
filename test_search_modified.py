        # phase 1. child network step (w)
        w_optim.zero_grad()
        logits = model(trn_X)
        loss = model.criterion(logits, trn_y)
        loss.backward()
        # gradient clipping
        nn.utils.clip_grad_norm_(model.weights(), config.w_grad_clip)

        """ yahan se meri lines"""
        rho = 0.05
        param_groups = w_optim.param_groups

        grad_norm = torch.norm(
          torch.stack([
            p.grad.norm(p=2)
            for group in param_groups for p in group["params"]
            if p.grad is not None
          ]),
          p = 2
        )

        for group in param_groups:
          scale = rho/(grad_norm + 1e-12) 

          for p in group["params"]:
            if p.grad is None: continue
            e_w = p.grad * scale
            p = p + e_w   #climb to the local maximum
            w_optim.state[p]["e_w"] = e_w

        w_optim.zero_grad()
        logits = model(trn_X)
        loss = model.criterion(logits, trn_y)
        loss.backward()

        param_groups = w_optim.param_groups
        for group in param_groups:
            for p in group["params"]:
                if p.grad is None: continue
                e_w = p.grad * scale
                w_optim.state[p]["e_w"] = e_w
                p = p - w_optim.state[p]["e_w"]  # get back to "w" from "w + e(w)"

        """" yahan tak meri lines """

        w_optim.step()

        prec1, prec5 = utils.accuracy(logits, trn_y, topk=(1, 5))
        losses.update(loss.item(), N)
        top1.update(prec1.item(), N)
        top5.update(prec5.item(), N)