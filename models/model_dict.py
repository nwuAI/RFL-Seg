from models.rfl_seg.build_sam_us import rfl_seg_model_registry

def get_model(modelname="SAM", args=None, opt=None):
    if modelname == "RFLSeg":
        model = rfl_seg_model_registry['vit_b'](args=args, checkpoint=args.sam_ckpt)
    else:
        raise RuntimeError("Could not find the model:", modelname)
    return model
