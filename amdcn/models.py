from baseline import BaselineModel
from parallel import ParallelDilationModel, ParallelAggregationModel, ParallelAggregationModelTest, ParallelAggregationModel4ColTest, ParallelAggregationModel3ColTest, ParallelAggregationModel2ColTest, ParallelAggregationModel1ColTest, ParallelNoAggregationModelTest, ParallelNoAggregationModel4ColTest, ParallelNoAggregationModel3ColTest, ParallelNoAggregationModel2ColTest, ParallelNoAggregationModel1ColTest

def add_arguments(parser):
    parser.add_argument('--model_type', default='baseline',
                        help='Model type (baseline, context)')
    parser.add_argument('--num_channels', type=int, default=32,
                        help='Number of channels in model')
    parser.add_argument('--num_layers', type=int, default=3,
                        help='Number of layers in model (baseline only)')
    return parser

def get_model(args):
    if args.model_type=='baseline':
        m = BaselineModel(num_layers=args.num_layers,num_channels=args.num_channels)
    elif args.model_type=='parallel':
        m = ParallelDilationModel(num_channels=args.num_channels)
    elif args.model_type=='parallel_aggregate':
        m = ParallelAggregationModel(num_channels=args.num_channels)
    elif args.model_type=='parallel_aggregate_test':
        m = ParallelAggregationModelTest(num_channels=args.num_channels)
    elif args.model_type=='parallel_aggregate_4_test':
        m = ParallelAggregationModel4ColTest(num_channels=args.num_channels)
    elif args.model_type=='parallel_aggregate_3_test':
        m = ParallelAggregationModel3ColTest(num_channels=args.num_channels)
    elif args.model_type=='parallel_aggregate_2_test':
        m = ParallelAggregationModel2ColTest(num_channels=args.num_channels)
    elif args.model_type=='parallel_aggregate_1_test':
        m = ParallelAggregationModel1ColTest(num_channels=args.num_channels)
    elif args.model_type=='parallel_noaggregate_test':
        m = ParallelNoAggregationModelTest(num_channels=args.num_channels)
    elif args.model_type=='parallel_noaggregate_4_test':
        m = ParallelNoAggregationModel4ColTest(num_channels=args.num_channels)
    elif args.model_type=='parallel_noaggregate_3_test':
        m = ParallelNoAggregationModel3ColTest(num_channels=args.num_channels)
    elif args.model_type=='parallel_noaggregate_2_test':
        m = ParallelNoAggregationModel2ColTest(num_channels=args.num_channels)
    elif args.model_type=='parallel_noaggregate_1_test':
        m = ParallelNoAggregationModel1ColTest(num_channels=args.num_channels)
    else:
        raise NameError('unknown model type: {}'.format(args.model_type))
    return m
