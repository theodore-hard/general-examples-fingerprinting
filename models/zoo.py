import torchvision

def get_net(modelname, pretrained=False, num_classes=1000):
    try:
        if pretrained:
            pretrained_model = eval('torchvision.models.{}'.format(modelname))(pretrained=True)
            if num_classes != 1000:
                model = eval('fingerprint.models.{}'.format(modelname))(num_classes=num_classes)
                #print('before copy weights:', model.state_dict())
                copy_weights_(pretrained_model.state_dict(), model.state_dict())
                #model.load_state_dict(dest_state_dict)
                #print('after copy weights:', model.state_dict())
                return model
            else:
                return pretrained_model
        else:
            try:
                model = eval('torchvision.models.{}'.format(modelname))(num_classes=num_classes)
            except Exception as e:
                model = eval('fingerprint.models.{}'.format(modelname))(num_classes=num_classes)
            return model
    except AssertionError as e:
        print(e)



def copy_weights_(src_state_dict, dst_state_dict):
    n_params = len(src_state_dict)
    n_success, n_skipped, n_shape_mismatch = 0, 0, 0

    for i, (src_param_name, src_param) in enumerate(src_state_dict.items()):
        if src_param_name in dst_state_dict:
            dst_param = dst_state_dict[src_param_name]
            if dst_param.data.shape == src_param.data.shape:
                dst_param.data.copy_(src_param.data)
                n_success += 1
            else:
                print('Mismatch: {} ({} != {})'.format(src_param_name, dst_param.data.shape, src_param.data.shape))
                n_shape_mismatch += 1
        else:
            n_skipped += 1
    print('=> # Success param blocks loaded = {}/{}, '
          '# Skipped = {}, # Shape-mismatch = {}'.format(n_success, n_params, n_skipped, n_shape_mismatch))
    #return dst_state_dict
