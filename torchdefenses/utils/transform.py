import copy
import logging


def transform_layer(input, from_inst, to_inst, args={}, attrs={}):
    if isinstance(input, from_inst):
        for key in args.keys():
            arg = args[key]
            if isinstance(arg, str):
                if arg.startswith("."):
                    args[key] = getattr(input, arg[1:])

        output = to_inst(**args)

        for key in attrs.keys():
            attr = attrs[key]
            if isinstance(attr, str):
                if attr.startswith("."):
                    attrs[key] = getattr(input, attr[1:])

            setattr(output, key, attrs[key])
    else:
        output = input
    return output


def transform_model(input, from_inst, to_inst, args={}, attrs={}, inplace=True, verbose=True):
    if inplace:
        output = input
        if verbose:
            logging.warning("The Input Model is CHANGED because inplace=True.")
    else:
        output = copy.deepcopy(input)

    if isinstance(output, from_inst):
        output = transform_layer(output, from_inst, to_inst, copy.deepcopy(args), copy.deepcopy(attrs))
    else:
        for name, module in output.named_children():
            setattr(output, name, transform_model(module, from_inst, to_inst, copy.deepcopy(args), copy.deepcopy(attrs), verbose=False))

    return output
