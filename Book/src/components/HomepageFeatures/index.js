import clsx from 'clsx';
import Heading from '@theme/Heading';
import styles from './styles.module.css';

const FeatureList = [
  {
    title: 'Master Physical AI',
    Svg: require('@site/static/img/undraw_docusaurus_mountain.svg').default,
    description: (
      <>
       This book provides a complete, structured pathway into the world of humanoid robotics. You will learn how intelligence moves from algorithms to real physical action—using ROS 2, Gazebo, Unity, NVIDIA Isaac, and Vision-Language-Action systems. Everything is designed so you can focus on building, simulating, and commanding embodied AI.
      </>
    ),
  },
  {
    title: 'Built for Serious Learners',
    Svg: require('@site/static/img/undraw_docusaurus_tree.svg').default,
    description: (
      <>
        Every chapter moves you from fundamentals to functional robotics systems. No fluff. No shortcuts. You will run real ROS 2 nodes, build humanoid models, simulate physics-accurate environments, generate perception data, and deploy natural-language robotic behaviors. All exercises are hands-on, reproducible, and aligned with industry-standard tools.
      </>
    ),
  },
  {
    title: 'From Simulation to Autonomy',
    Svg: require('@site/static/img/undraw_docusaurus_react.svg').default,
    description: (
      <>
        The book connects high-level AI reasoning to low-level motor control through modern robotics platforms. You will design a digital twin, train advanced perception systems, integrate VLA pipelines, and ultimately build a fully autonomous humanoid capable of completing real-world tasks from a single voice command.
      </>
    ),
  },
];

function Feature({Svg, title, description}) {
  return (
    <div className={clsx('col col--4')}>
      <div className="text--center">
        <Svg className={styles.featureSvg} role="img" />
      </div>
      <div className="text--center padding-horiz--md">
        <Heading as="h3">{title}</Heading>
        <p>{description}</p>
      </div>
    </div>
  );
}

export default function HomepageFeatures() {
  return (
    <section className={styles.features}>
      <div className="container">
        <div className="row">
          {FeatureList.map((props, idx) => (
            <Feature key={idx} {...props} />
          ))}
        </div>
      </div>
    </section>
  );
}
